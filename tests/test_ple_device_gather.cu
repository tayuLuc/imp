// Device-side n-gram gather must return the same bytes as the host reader, on the same ids.
//
// This is a GPU test and CI cannot run it: ple-ci.yml builds and runs the CPU suites, and the
// fork has no self-hosted GPU runner. So it is built there as an artifact and executed on the
// 5090 node - build in CI, measure on hardware. It is built against the same synthetic table the
// host suite uses, so a failure here is a real disagreement and not a property of some 47.7 GiB
// checkpoint we would have to download to reproduce it.
//
// What it proves: the kernel returns the same bytes as the host reader eagerly, and the same bytes
// again when replayed from a captured CUDA graph (twice, so a replay-only divergence would show).
// That is what lets it sit inside a graph. What it still does not prove: anything about a cold page
// cache on the real 47.7 GiB checkpoint, or the cost of an HMM software fault. Both need hardware.

#include "compute/ple.h"
#include "model/ngram_table.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int kHeadDim = 160;
constexpr int kRowsPerShard = 4096;
constexpr int kShards = 2;
constexpr int kNHeads = 4;
constexpr int kEos = 99;

#define CHECK_CUDA(expr)                                                                 \
    do {                                                                                 \
        const cudaError_t err_ = (expr);                                                 \
        if (err_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,            \
                         cudaGetErrorString(err_));                                      \
            return 2;                                                                    \
        }                                                                                \
    } while (0)

// The loader checks dtype and shape, not values; this fills every FP8 byte class so the kernel
// meets subnormals, NaN and -0.0 rather than only ordinary values.
uint8_t f8(int row, int j) { return static_cast<uint8_t>((row * 31 + j * 7 + 11) & 0xff); }

struct Fixture {
    std::string dir;
    ~Fixture() {
        if (dir.empty())
            return;
        for (const char* f : {"/model.safetensors.index.json", "/model-00001-of-00002.safetensors"})
            std::remove((dir + f).c_str());
        rmdir(dir.c_str());
    }
};

// Writes a two-shard table shaped exactly like the real one, then returns the open NGramTable.
std::unique_ptr<imp::NGramTable> make_table(Fixture& fx, std::vector<std::vector<uint8_t>>& bytes) {
    char tmpl[] = "/tmp/imp_ple_dev_XXXXXX";
    if (!mkdtemp(tmpl))
        return nullptr;
    fx.dir = tmpl;
    const std::string index_path = fx.dir + "/model.safetensors.index.json";
    std::string index = "{\"metadata\":{\"total_size\":0},\"weight_map\":{";
    for (int k = 0; k < kShards; k++)
        index += (k ? "," : "") + std::string("\"ngram_embedding.shard_") + std::to_string(k) +
                 ".weight\":\"model-00001-of-00002.safetensors\"";
    index += "}}";
    FILE* f = std::fopen(index_path.c_str(), "wb");
    if (!f)
        return nullptr;
    std::fwrite(index.data(), 1, index.size(), f);
    std::fclose(f);

    const size_t header = 1024;
    const size_t tensor_bytes = static_cast<size_t>(kRowsPerShard) * kHeadDim;
    bytes.assign(kShards, std::vector<uint8_t>(tensor_bytes));
    const std::string shard_path = fx.dir + "/model-00001-of-00002.safetensors";
    f = std::fopen(shard_path.c_str(), "wb");
    if (!f)
        return nullptr;
    std::vector<uint8_t> pad(header, 0);
    std::fwrite(pad.data(), 1, pad.size(), f);
    for (int k = 0; k < kShards; k++) {
        for (size_t i = 0; i < tensor_bytes; i++)
            bytes[k][i] = f8(static_cast<int>(i / kHeadDim) + k * 7, static_cast<int>(i % kHeadDim));
        std::fwrite(bytes[k].data(), 1, tensor_bytes, f);
    }
    std::fclose(f);
    return imp::NGramTable::open(fx.dir, 1, kEos);
}

int fail(const char* what, std::size_t slot, const uint16_t* host, const uint16_t* dev) {
    std::fprintf(stderr, "FAIL %s at slot %zu: host=0x%04x device=0x%04x\n", what, slot, host[slot],
                 dev[slot]);
    return 1;
}

}  // namespace

int main() {
    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        std::fprintf(stderr, "no CUDA device; run this on the 5090 node\n");
        return 2;
    }
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    // The attribute that decides whether this whole approach can work: can the device dereference
    // host virtual addresses at all? If this is 0 the kernel faults regardless of how correct it is.
    int pageable_access = 0;
    const cudaError_t attr_err =
        cudaDeviceGetAttribute(&pageable_access, cudaDevAttrPageableMemoryAccess, 0);
    std::printf("device: %s, sm_%d%d, pageable-memory-access=%d%s\n", prop.name, prop.major, prop.minor,
                attr_err == cudaSuccess ? pageable_access : -1,
                attr_err == cudaSuccess ? "" : " (attribute unavailable)");

    Fixture fx;
    std::vector<std::vector<uint8_t>> bytes;
    const std::unique_ptr<imp::NGramTable> table = make_table(fx, bytes);
    if (table == nullptr) {
        std::fprintf(stderr, "FAIL could not open the synthetic table\n");
        return 2;
    }
    if (table->empty() || table->map_base() == nullptr) {
        std::fprintf(stderr, "FAIL table is not mapped; device gather needs the mmap\n");
        return 2;
    }

    const int head_dim = table->head_dim();
    const int n_heads = table->n_heads();
    const size_t rows = table->shard_count();
    std::printf("table: %lld rows x %d, %d heads, %zu shards, scale %.6g\n",
                static_cast<long long>(table->total_rows()), head_dim, n_heads, rows, table->scale());

    // Ids that cross a shard boundary, repeat, hit the very first and very last row, and go out of
    // range both ways - every branch the kernel has.
    std::vector<int64_t> ids;
    ids.push_back(0);
    ids.push_back(kRowsPerShard - 1);
    ids.push_back(kRowsPerShard);  // first row of shard 1
    ids.push_back(kRowsPerShard + 13);
    ids.push_back(table->total_rows() - 1);
    ids.push_back(17);
    ids.push_back(17);  // repeated
    ids.push_back(-1);  // out of range low
    ids.push_back(table->total_rows());  // out of range high
    const size_t count = ids.size();

    std::vector<uint16_t> host(count * head_dim, 0xbeef);
    table->gather(ids.data(), static_cast<int>(count), host.data());

    void* d_ids = nullptr;
    void* d_starts = nullptr;
    void* d_offs = nullptr;
    void* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ids, count * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_starts, rows * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_offs, rows * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_out, count * head_dim * sizeof(uint16_t)));
    // 0xcd rather than zero: an untouched slot must not read as a valid row.
    CHECK_CUDA(cudaMemset(d_out, 0xcd, count * head_dim * sizeof(uint16_t)));
    CHECK_CUDA(cudaMemcpy(d_ids, ids.data(), count * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_starts, table->shard_row_starts(), rows * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offs, table->shard_byte_offs(), rows * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    imp::ple_gather(static_cast<const int64_t*>(d_ids), static_cast<uint32_t>(count), table->map_base(),
                    static_cast<const int64_t*>(d_starts), static_cast<const uint64_t*>(d_offs),
                    static_cast<int>(rows), table->total_rows(), head_dim, table->scale(),
                    static_cast<uint16_t*>(d_out), nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint16_t> dev(count * head_dim, 0);
    CHECK_CUDA(cudaMemcpy(dev.data(), d_out, dev.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    size_t mismatches = 0;
    for (size_t i = 0; i < count * static_cast<size_t>(head_dim); i++) {
        if (host[i] == dev[i])
            continue;
        if (mismatches < 5)
            std::fprintf(stderr, "FAIL row %zu col %zu id %lld: host=0x%04x device=0x%04x\n",
                         i / head_dim, i % head_dim, static_cast<long long>(ids[i / head_dim]), host[i],
                         dev[i]);
        mismatches++;
    }
    std::printf("compared %zu halves across %zu rows: %zu mismatches\n", count * head_dim, count,
                mismatches);
    if (mismatches != 0) {
        std::fprintf(stderr, "FAIL device gather differs from the host reader\n");
        return 1;
    }
    std::printf("PASS device gather is byte-identical to the host reader\n");

    // The point of putting this kernel inside a CUDA graph is that a graph can replay it, so
    // replay is what has to be tested: capture is not enough, and neither is a second eager
    // launch. A graph bakes the launch geometry, so a kernel that is correct eagerly can still be
    // wrong under replay if it depends on anything the capture did not record.
    //
    // Stream capture is refused on the legacy default stream, which is why the first pass above
    // passed nullptr. Capturing therefore needs a stream of its own.
    cudaStream_t cap = nullptr;
    CHECK_CUDA(cudaStreamCreate(&cap));
    CHECK_CUDA(cudaMemsetAsync(d_out, 0xcd, count * head_dim * sizeof(uint16_t), cap));
    CHECK_CUDA(cudaStreamBeginCapture(cap, cudaStreamCaptureModeThreadLocal));
    imp::ple_gather(static_cast<const int64_t*>(d_ids), static_cast<uint32_t>(count), table->map_base(),
                    static_cast<const int64_t*>(d_starts), static_cast<const uint64_t*>(d_offs),
                    static_cast<int>(rows), table->total_rows(), head_dim, table->scale(),
                    static_cast<uint16_t*>(d_out), cap);
    cudaGraph_t graph = nullptr;
    const cudaError_t end_err = cudaStreamEndCapture(cap, &graph);
    if (end_err != cudaSuccess || graph == nullptr) {
        std::fprintf(stderr, "FAIL capture did not produce a graph: %s\n", cudaGetErrorString(end_err));
        return 2;
    }
    cudaGraphExec_t exec = nullptr;
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    std::printf("captured and instantiated a graph\n");

    // Replay twice. The first launch of a graph is allowed to differ from subsequent ones, and a
    // one-shot check would miss a bug that only shows on the second pass.
    for (int pass = 0; pass < 2; ++pass) {
        CHECK_CUDA(cudaMemsetAsync(d_out, 0xcd, count * head_dim * sizeof(uint16_t), cap));
        CHECK_CUDA(cudaGraphLaunch(exec, cap));
        CHECK_CUDA(cudaStreamSynchronize(cap));
        CHECK_CUDA(cudaMemcpy(dev.data(), d_out, dev.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < count * static_cast<size_t>(head_dim); i++)
            if (host[i] != dev[i])
                ++bad;
        std::printf("replay %d: %zu halves, %zu mismatches\n", pass + 1, count * head_dim, bad);
        if (bad != 0) {
            std::fprintf(stderr, "FAIL replay %d differs from the host reader\n", pass + 1);
            return 1;
        }
    }
    std::printf("PASS the captured graph replays byte-identically to the host reader\n");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(cap);
    return 0;
}