// The three PLE table readers must return the same rows: "mmap" faults pages in,
// "pread" and "uring" read the selected rows into staging. CPU lane, no CUDA, on
// a synthetic SafeTensors shard file.

#include <gtest/gtest.h>

#include "core/process_diag.h"
#include "model/ngram_table.h"

#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace imp;

namespace {

constexpr int kHeadDim = 160;
constexpr int64_t kHeadVocab = 2048;
constexpr float kScale = 1.0f / 32768.0f;  // bf16 0x3800 = 2^-15
constexpr int kRowsPerShard = 4096;
constexpr int kShards = 2;
constexpr int kNHeads = 4;
constexpr int kNGramSize = 3;
constexpr int kEos = 99;

struct TableFixture {
    std::string dir;
    std::string shard_file;
    std::vector<std::vector<uint8_t>> shard_bytes;

    ~TableFixture() {
        if (!dir.empty())
            for (const char* f : {"/model.safetensors.index.json", "/model-00001-of-00002.safetensors"})
                std::remove((dir + f).c_str());
        if (!dir.empty())
            rmdir(dir.c_str());
    }
};

// The loader checks dtype and shape, not values.
uint8_t f8(int row, int j) { return static_cast<uint8_t>((row * 31 + j * 7 + 11) & 0x7b); }

// FP8 E4M3 decode from the spec, the same derivation the quantise ref test uses,
// so this test does not compare the table against a copy of its own maths.
float f8_ref(uint8_t b) {
    const uint32_t sign = (b >> 7) & 1;
    const uint32_t exp = (b >> 3) & 0x0f;
    const uint32_t man = b & 0x07;
    const float mag = exp == 0 ? static_cast<float>(man) / 512.0f
                               : (1.0f + static_cast<float>(man) / 8.0f) * std::ldexp(1.0f, exp - 7);
    return sign ? -mag : mag;
}

float f16_ref(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t man = h & 0x3ff;
    if (exp == 0)
        return sign ? -std::ldexp(static_cast<float>(man), -24) : std::ldexp(static_cast<float>(man), -24);
    if (exp == 0x1f)
        return man ? NAN : (sign ? -INFINITY : INFINITY);
    const float mag = std::ldexp(1.0f + static_cast<float>(man) / 1024.0f, static_cast<int>(exp) - 15);
    return sign ? -mag : mag;
}

TableFixture make_table() {
    TableFixture fx;
    char tmpl[] = "/tmp/imp_ngram_test_XXXXXX";
    const char* d = mkdtemp(tmpl);
    EXPECT_NE(d, nullptr);
    fx.dir = d;

    const std::string prefix = ".layers.1.ple.ple_embedding.";
    std::vector<std::string> names;
    std::vector<std::string> hdr;
    size_t off = 0;
    // I64 hash buffers, then the F8 shards, then the BF16 scale.
    const std::vector<std::pair<std::string, std::vector<int64_t>>> i64 = {
        {"layer_multipliers", {7, 11, 13}},
        {"ngram_heads_offsets", {0, kHeadVocab, 2 * kHeadVocab, 3 * kHeadVocab}},
        {"ngram_heads_vocab_sizes", {kHeadVocab, kHeadVocab, kHeadVocab, kHeadVocab}},
    };
    for (const auto& [name, vals] : i64) {
        hdr.push_back("\"" + prefix + name + "\":{\"dtype\":\"I64\",\"shape\":[" + std::to_string(vals.size()) +
                      "],\"data_offsets\":[" + std::to_string(off) + "," +
                      std::to_string(off + vals.size() * 8) + "]}");
        off += vals.size() * 8;
    }
    for (int k = 0; k < kShards; ++k) {
        const std::string name = "ngram_embedding.shard_" + std::to_string(k) + ".weight";
        hdr.push_back("\"" + prefix + name + "\":{\"dtype\":\"F8_E4M3\",\"shape\":[" + std::to_string(kRowsPerShard) +
                      "," + std::to_string(kHeadDim) + "],\"data_offsets\":[" + std::to_string(off) + "," +
                      std::to_string(off + kRowsPerShard * kHeadDim) + "]}");
        fx.shard_bytes.emplace_back(kRowsPerShard * kHeadDim);
        for (int r = 0; r < kRowsPerShard; ++r)
            for (int j = 0; j < kHeadDim; ++j)
                fx.shard_bytes[k][static_cast<size_t>(r) * kHeadDim + j] = f8(k * kRowsPerShard + r, j);
        off += kRowsPerShard * kHeadDim;
        names.push_back(name);
    }
    hdr.push_back("\"" + prefix + "ngram_embedding.weight_scale\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[" +
                  std::to_string(off) + "," + std::to_string(off + 2) + "]}");

    std::string head = "{\"__metadata__\":{},";
    for (size_t i = 0; i < hdr.size(); ++i)
        head += (i ? "," : "") + hdr[i];
    head += "}";
    while ((head.size() % 8) != 0)
        head += " ";

    fx.shard_file = fx.dir + "/model-00001-of-00002.safetensors";
    const int fd = ::open(fx.shard_file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    EXPECT_GE(fd, 0);
    const uint64_t hlen = head.size();
    EXPECT_EQ(write(fd, &hlen, 8), 8);
    EXPECT_EQ(write(fd, head.data(), static_cast<ssize_t>(head.size())), static_cast<ssize_t>(head.size()));
    size_t cur = 0;
    for (const auto& t : i64) {
        EXPECT_EQ(write(fd, t.second.data(), static_cast<ssize_t>(t.second.size() * 8)),
                  static_cast<ssize_t>(t.second.size() * 8));
        cur += t.second.size() * 8;
    }
    for (int k = 0; k < kShards; ++k) {
        EXPECT_EQ(write(fd, fx.shard_bytes[k].data(), static_cast<ssize_t>(fx.shard_bytes[k].size())),
                  static_cast<ssize_t>(fx.shard_bytes[k].size()));
        cur += fx.shard_bytes[k].size();
    }
    const uint16_t scale = 0x3800;  // 2^-15 in bf16
    EXPECT_EQ(write(fd, &scale, 2), 2);
    ::close(fd);
    EXPECT_EQ(cur, off);  // ASSERT_ cannot be used here: make_table returns a value

    std::string wm = "{\"weight_map\":{";
    const std::string file = "model-00001-of-00002.safetensors";
    std::vector<std::string> tensors;
    for (const auto& [name, vals] : i64)
        tensors.push_back(name);
    for (const auto& name : names)
        tensors.push_back(name);
    tensors.push_back("ngram_embedding.weight_scale");
    for (size_t i = 0; i < tensors.size(); ++i)
        wm += (i ? "," : "") + ("\"" + prefix + tensors[i] + "\":\"" + file + "\"");
    wm += "}}";
    const std::string idx = fx.dir + "/model.safetensors.index.json";
    const int ifd = ::open(idx.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    EXPECT_GE(ifd, 0);
    EXPECT_EQ(write(ifd, wm.data(), static_cast<ssize_t>(wm.size())), static_cast<ssize_t>(wm.size()));
    ::close(ifd);
    return fx;
}

std::vector<int64_t> ids_for(const TableFixture& fx, int tokens, uint64_t seed) {
    std::vector<int64_t> ids(static_cast<size_t>(tokens) * kNHeads);
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < kNHeads; ++h) {
            const int64_t vocab = kHeadVocab;
            const int64_t off = h * kHeadVocab;
            ids[static_cast<size_t>(t) * kNHeads + h] =
                off + static_cast<int64_t>((seed * 6364136223846793005ull + (t * 7 + h) * 2654435761ull) %
                                           static_cast<uint64_t>(vocab));
        }
    (void)fx;
    return ids;
}

std::unique_ptr<ProcessDiag> set_backend(const std::string& name, int coalesce_kib = 4) {
    auto d = std::make_unique<ProcessDiag>();
    d->ple_table_backend = name;
    d->ple_io_threads = 2;
    d->ple_coalesce_kib = coalesce_kib;
    d->ple_queue_depth = 8;
    d->ple_log_stats = false;
    process_diag_set(*d);
    return d;  // keep the override alive for the test body
}

}  // namespace

TEST(NGramTableBackends, EveryBackendGathersTheSameRows) {
    const TableFixture fx = make_table();
    const std::vector<int64_t> ids = ids_for(fx, 37, 7);
    const size_t count = ids.size();
    const size_t out_elems = count * kHeadDim;
    const float scale = kScale;

    // What gather() owes every slot: the F8 row behind its id, times the table scale.
    std::vector<float> expected(out_elems);
    for (size_t i = 0; i < count; ++i) {
        const int64_t id = ids[i];
        ASSERT_GE(id, 0) << "fixture bug";
        ASSERT_LT(id, kShards * kRowsPerShard) << "fixture bug";
        const uint8_t* row = fx.shard_bytes[id / kRowsPerShard].data() + (id % kRowsPerShard) * kHeadDim;
        for (int j = 0; j < kHeadDim; ++j)
            expected[i * kHeadDim + j] = f8_ref(row[j]) * scale;
    }

    std::vector<std::vector<uint16_t>> results;
    for (const char* backend : {"mmap", "pread", "uring"}) {
        auto guard = set_backend(backend);
        auto t = NGramTable::open(fx.dir, 1, kEos);
        ASSERT_NE(t, nullptr) << backend;
        EXPECT_EQ(t->n_heads(), kNHeads);
        EXPECT_EQ(t->head_dim(), kHeadDim);
        EXPECT_EQ(t->total_rows(), kShards * kRowsPerShard);
        std::vector<uint16_t> out(out_elems, 0xbeef);
        t->gather(ids.data(), static_cast<int>(count / kNHeads), out.data());
        for (size_t i = 0; i < out_elems; ++i) {
            // fp16 keeps ~3 decimal digits; 1e-3 relative is below its own error.
            ASSERT_NEAR(f16_ref(out[i]), expected[i], std::fabs(expected[i]) * 1e-3 + 1e-9)
                << backend << " slot " << i / kHeadDim << " col " << i % kHeadDim << " id " << ids[i / kHeadDim];
        }
        results.push_back(std::move(out));
        if (std::string(backend) == "mmap")
            EXPECT_EQ(t->backend(), NGramBackend::Mmap);
        else
            EXPECT_NE(t->backend(), NGramBackend::Mmap) << "streaming backend downgraded";
    }
    EXPECT_EQ(results[0], results[1]);
    EXPECT_EQ(results[0], results[2]);
}

TEST(NGramTableBackends, EveryBackendAccountsForWhatItSelected) {
    // Regression: gather_mmap_ incremented no counters, so a run on the default backend
    // logged zeros and could not be compared against a pread run. rows and bytes must
    // now move for every backend; ranges stays 0 for mmap because the kernel, not a
    // read call, is what resolves an address there.
    const TableFixture fx = make_table();
    std::vector<int64_t> ids(kNHeads, 42);
    for (const char* backend : {"mmap", "pread"}) {
        auto guard = set_backend(backend);
        auto t = NGramTable::open(fx.dir, 1, kEos);
        ASSERT_NE(t, nullptr) << backend;
        std::vector<uint16_t> out(kNHeads * kHeadDim, 0);
        t->gather(ids.data(), 1, out.data());
        const NGramTable::Stats s = t->stats();
        // rows is slots, so one token x kNHeads heads, and that is uniform across backends -
        // this is the equality the counters exist for.
        EXPECT_EQ(s.gathers, 1u) << backend;
        EXPECT_EQ(s.rows, static_cast<uint64_t>(kNHeads)) << backend;
        // bytes_read is deliberately NOT uniform: it is what the backend moved. Four heads
        // pointing at one row is one row's worth of bytes through pread and four slots'
        // worth through mmap, which issues no read at all. Pin the asymmetry instead of
        // demanding an equality that would misreport one of the two.
        if (std::string(backend) == "pread") {
            EXPECT_GE(s.ranges, 1u) << backend;
            EXPECT_EQ(s.bytes_read, static_cast<uint64_t>(kHeadDim)) << backend;
        } else {
            EXPECT_EQ(s.ranges, 0u) << backend;
            EXPECT_EQ(s.bytes_read, static_cast<uint64_t>(kNHeads) * kHeadDim) << backend;
        }
    }
}

TEST(NGramTableBackends, RepeatedRowsAreReadOnce) {
    const TableFixture fx = make_table();
    // One token, every head pointing at the same row: 4 slots, 1 unique row.
    std::vector<int64_t> ids(kNHeads, 42);
    auto guard = set_backend("pread");
    auto t = NGramTable::open(fx.dir, 1, kEos);
    ASSERT_NE(t, nullptr);
    std::vector<uint16_t> out(kNHeads * kHeadDim, 0);
    t->gather(ids.data(), 1, out.data());
    // Read once means one range, not one row: rows counts the 4 slots the caller asked for,
    // ranges counts the unique row pread actually fetched.
    EXPECT_EQ(t->stats().rows, static_cast<uint64_t>(kNHeads));
    EXPECT_EQ(t->stats().ranges, 1u);
    EXPECT_EQ(t->stats().gathers, 1u);
    // Every head's slot holds the same converted row.
    for (int h = 1; h < kNHeads; ++h)
        for (int j = 0; j < kHeadDim; ++j)
            EXPECT_EQ(out[static_cast<size_t>(h) * kHeadDim + j], out[j]) << "head " << h << " col " << j;
}

TEST(NGramTableBackends, OutOfRangeIdsAreZeroed) {
    const TableFixture fx = make_table();
    auto guard = set_backend("pread");
    auto t = NGramTable::open(fx.dir, 1, kEos);
    ASSERT_NE(t, nullptr);
    std::vector<int64_t> ids(kNHeads, 42);
    ids[0] = 0;
    ids[1] = kShards * kRowsPerShard;  // one past the last row
    std::vector<uint16_t> out(kNHeads * kHeadDim, 0xbeef);
    t->gather(ids.data(), 1, out.data());
    for (int h = 0; h < kNHeads; ++h) {
        if (h == 1) {
            for (int j = 0; j < kHeadDim; ++j)
                EXPECT_EQ(out[static_cast<size_t>(h) * kHeadDim + j], 0) << "out-of-range slot must be zero";
            continue;
        }
        // Compare against the row itself: a fixture byte can be 0x00, and the
        // converted value is then legitimately zero.
        const uint8_t* row = fx.shard_bytes[ids[h] / kRowsPerShard].data() + (ids[h] % kRowsPerShard) * kHeadDim;
        for (int j = 0; j < kHeadDim; ++j) {
            const float want = f8_ref(row[j]) * kScale;
            ASSERT_NEAR(f16_ref(out[static_cast<size_t>(h) * kHeadDim + j]), want,
                        std::fabs(want) * 1e-3 + 1e-9)
                << "head " << h << " col " << j << " id " << ids[h];
        }
    }
}

TEST(NGramTableBackends, UnknownBackendNameFallsBackToMmap) {
    const TableFixture fx = make_table();
    auto guard = set_backend("prea");
    auto t = NGramTable::open(fx.dir, 1, kEos);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->backend(), NGramBackend::Mmap);
}

TEST(NGramTableBackends, NameRoundTrip) {
    EXPECT_EQ(ngram_backend_from_string("pread"), NGramBackend::Pread);
    EXPECT_EQ(ngram_backend_from_string("uring"), NGramBackend::Uring);
    EXPECT_EQ(ngram_backend_from_string("mmap"), NGramBackend::Mmap);
    EXPECT_STREQ(ngram_backend_name(NGramBackend::Pread), "pread");
    EXPECT_STREQ(ngram_backend_name(NGramBackend::Uring), "uring");
    EXPECT_STREQ(ngram_backend_name(NGramBackend::Mmap), "mmap");
}

TEST(NGramTableBackends, BatchedGatherEqualsOneRowAtATime) {
    // The oracle is gather() itself. A single-row gather never coalesces, so it has
    // no holes in its layout; a batched one does. Any arithmetic that assumes rows
    // inside a coalesced range are adjacent shows up here and nowhere cheaper.
    //
    // The window has to be wide: at 4 KiB and 36 scattered ids almost nothing
    // merges, no hole appears, and the property holds vacuously. A perturbation
    // run (the same bug reintroduced) is what caught that.
    const TableFixture fx = make_table();
    std::vector<int64_t> ids = ids_for(fx, 9, 11);  // scattered: gaps between neighbours
    for (const char* backend : {"mmap", "pread", "uring"}) {
        auto guard = set_backend(backend, 256);
        auto batched = NGramTable::open(fx.dir, 1, kEos);
        ASSERT_NE(batched, nullptr) << backend;
        std::vector<uint16_t> got(ids.size() * kHeadDim, 0);
        batched->gather(ids.data(), static_cast<int>(ids.size() / kNHeads), got.data());

        const int tokens = static_cast<int>(ids.size() / kNHeads);
        for (int t = 0; t < tokens; ++t) {
            std::vector<uint16_t> one(kNHeads * kHeadDim, 0);
            batched->gather(&ids[static_cast<size_t>(t) * kNHeads], 1, one.data());
            for (size_t i = 0; i < one.size(); ++i) {
                ASSERT_EQ(got[static_cast<size_t>(t) * kNHeads * kHeadDim + i], one[i])
                    << backend << " token " << t << " elem " << i;
            }
        }
    }
}

TEST(NGramTableBackends, OutputDoesNotDependOnTheCoalescingWindow) {
    // Same rows, different read-ahead: a wider window merges neighbours the narrow
    // one keeps apart, so a layout bug that only appears with holes flips the result.
    const TableFixture fx = make_table();
    const std::vector<int64_t> ids = ids_for(fx, 23, 5);
    const int tokens = static_cast<int>(ids.size() / kNHeads);
    for (const char* backend : {"pread", "uring"}) {
        std::vector<uint16_t> narrow, wide;
        {
            auto guard = set_backend(backend, 1);
            auto t = NGramTable::open(fx.dir, 1, kEos);
            ASSERT_NE(t, nullptr) << backend;
            narrow.assign(ids.size() * kHeadDim, 0);
            t->gather(ids.data(), tokens, narrow.data());
        }
        {
            auto guard = set_backend(backend, 256);
            auto t = NGramTable::open(fx.dir, 1, kEos);
            ASSERT_NE(t, nullptr) << backend;
            wide.assign(ids.size() * kHeadDim, 0);
            t->gather(ids.data(), tokens, wide.data());
        }
        ASSERT_EQ(narrow, wide) << backend << ": the coalescing window changed the rows";
    }
}
