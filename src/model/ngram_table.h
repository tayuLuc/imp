#pragma once
// Qwen4Exp PLE n-gram table: the F8_E4M3 embedding shards live in their SafeTensors
// file (320001536 rows x 160 B = 47.7 GiB on this checkpoint, never fully resident;
// upstream says "51 GiB", which is the same bytes counted in decimal GB), the I64
// hash buffers come from the main shards. hash() reproduces Qwen4ExpTextNGramEmbedding.forward,
// gather() the embedding lookup.
//
// Three readers (NGramBackend): "mmap" maps the file and lets the page cache own
// residency, "pread" and "uring" read the selected rows into host staging. All three
// produce identical rows; they differ in who faults the pages in.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace imp {

class HostTaskPool;

// Pure hash, testable without a checkpoint. `ctx` holds ngram_size-1 tokens before tokens[0]
// (EOS-filled at a sequence start); a shift never crosses an EOS. ids_out is [n, n_heads].
struct NGramHashParams {
    const int64_t* multipliers;  // [ngram_size]
    const int64_t* offsets;      // [n_heads]
    const int64_t* vocab_sizes;  // [n_heads]
    int ngram_size;
    int n_heads;  // (ngram_size - 1) * heads_per_ngram
    int eos_token_id;
};
void ngram_hash(const NGramHashParams& p, const int32_t* ctx, const int32_t* tokens, int n, int64_t* ids_out);

// Context of the token at position pos0: ctx[i] = seq[pos0 - ctx_len + i], eos before
// position 0, where seq = in[0, n_in) then out[0, n_out). False (ctx eos-filled) when a
// needed position lies past the history.
[[nodiscard]] inline bool ngram_context_at(const int32_t* in, int n_in, const int32_t* out, int n_out, int pos0,
                             int ctx_len, int32_t eos, int32_t* ctx) {
    bool ok = true;
    for (int i = 0; i < ctx_len; i++) {
        int p = pos0 - ctx_len + i;
        if (p < 0)
            ctx[i] = eos;
        else if (p < n_in && in != nullptr)
            ctx[i] = in[p];
        else if (p - n_in < n_out && out != nullptr && p >= n_in)
            ctx[i] = out[p - n_in];
        else {
            ctx[i] = eos;
            ok = false;
        }
    }
    return ok;
}

enum class NGramBackend { Mmap, Pread, Uring };

// Parses ple.table_backend. Unknown spellings fall back to "mmap" (logged) so a
// typo never turns into a refusal at load.
[[nodiscard]] NGramBackend ngram_backend_from_string(const std::string& name);
const char* ngram_backend_name(NGramBackend b);

class NGramTable {
public:
    // Row [id] as a byte range in the shard file.
    struct RowRef {
        uint64_t off;
        int64_t id;
    };
    // One read: [file_off, file_off + len) into staging[buf_off, buf_off + len).
    struct Range {
        uint64_t file_off;
        size_t buf_off;
        size_t len;
    };

    ~NGramTable();
    NGramTable(const NGramTable&) = delete;
    NGramTable& operator=(const NGramTable&) = delete;

    // Resolves every `.layers.<layer>.ple.ple_embedding.*` tensor through the sharded index.
    // Null on any shortfall: without its table the model cannot be served.
    static std::unique_ptr<NGramTable> open(const std::string& model_dir, int layer, int eos_token_id);

    int n_heads() const { return n_heads_; }
    int head_dim() const { return head_dim_; }
    int embed_dim() const { return n_heads_ * head_dim_; }
    int ngram_size() const { return ngram_size_; }
    int context_len() const { return ngram_size_ - 1; }
    int64_t total_rows() const { return total_rows_; }
    float scale() const { return scale_; }
    NGramBackend backend() const { return backend_; }

    void hash(const int32_t* ctx, const int32_t* tokens, int n, int64_t* ids_out) const;
    // out [n, embed_dim] FP16 bits = F8 row * scale. Byte-identical for every backend.
    void gather(const int64_t* ids, int n, uint16_t* out) const;

    struct Stats {
        uint64_t gathers = 0;
        uint64_t rows = 0;
        uint64_t ranges = 0;
        uint64_t bytes_read = 0;
        uint64_t read_errors = 0;
        uint64_t direct_reads = 0;
        uint64_t buffered_reads = 0;
        uint64_t uring_submits = 0;
    };
    [[nodiscard]] Stats stats() const;

private:
    NGramTable() = default;
    struct Shard {
        int64_t row_start;
        int64_t rows;
        const uint8_t* data;   // mmap view (nullptr when not mapped)
        uint64_t byte_off;     // file offset of row 0, for the streaming readers
    };

    // One read of plan_ into staging_, through the chosen backend.
    bool stream_() const;

    // Where a range is read from and into, widened and aligned for O_DIRECT.
    struct Span {
        uint64_t file_off;
        uint64_t file_len;
        size_t buf_off;
        size_t buf_len;
        size_t first_row;  // index into plan_
        size_t rows;
    };
    // Both return false on any short or failed read.
    bool read_pread_(const std::vector<Span>& spans) const;
    bool read_uring_(const std::vector<Span>& spans) const;
    void free_staging_();

    void init_backend_();
    void log_backend_() const;
    void gather_mmap_(const int64_t* ids, size_t count, uint16_t* out) const;
    bool gather_streaming_(const int64_t* ids, size_t count, uint16_t* out) const;

    // A gather of 16 heads over 65k tokens is 1M rows, 160 MiB.
    static constexpr size_t kMaxStaging = 512ull << 20;

    void* map_ = nullptr;
    size_t map_size_ = 0;
    std::vector<Shard> shards_;
    int64_t total_rows_ = 0;
    std::vector<int64_t> multipliers_, offsets_, vocab_sizes_;
    float scale_ = 1.0f;
    int n_heads_ = 0;
    int head_dim_ = 0;
    int ngram_size_ = 0;
    int eos_ = -1;
    mutable int64_t n_out_of_range_ = 0;

    NGramBackend backend_ = NGramBackend::Mmap;
    std::string backend_requested_ = "mmap";
    std::string shard_path_;
    int fd_ = -1;         // O_RDONLY, streaming readers
    int fd_direct_ = -1;  // O_RDONLY|O_DIRECT, -1 when the filesystem refuses
    int io_threads_ = 0;
    size_t coalesce_bytes_ = 0;
    int queue_depth_ = 0;
    bool log_stats_ = true;
    std::unique_ptr<HostTaskPool> pool_;
    // The ring lives in a heap block sized at init, so this header needs no liburing.
    void* ring_mem_ = nullptr;   // the struct itself
    bool ring_live_ = false;     // queue_init succeeded, queue_exit is owed

    // Scratch for one gather. Reused across calls, so one gather at a time per
    // table (io_mu_), which is also the order the host caller runs them in.
    mutable std::mutex io_mu_;
    mutable std::vector<RowRef> plan_;      // unique rows, sorted by file offset
    mutable std::vector<uint32_t> plan_ix_;  // plan slot per (token, head); kNoPlan when invalid
    mutable std::vector<uint8_t*> row_ptr_;  // staging address of each plan row
    mutable uint8_t* staging_ = nullptr;     // kAlign-aligned, staging_cap_ bytes
    mutable size_t staging_cap_ = 0;
    mutable Stats stats_;
    mutable std::mutex stats_mu_;
};

}  // namespace imp
