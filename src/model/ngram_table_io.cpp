// Streaming readers for the Qwen4Exp PLE n-gram table: "pread" and "uring" move the
// selected rows into host staging, "mmap" (the default) lives in ngram_table.cpp and
// lets the page cache own residency. Same bytes either way.
//
// Staging is 4 KiB-aligned and O_DIRECT spans are widened to 4 KiB at both ends: a
// 160 B row run is essentially never aligned, and an unaligned read would land in
// the page cache, which is the thing these backends exist to avoid. The widened
// bytes are read, never used.

#include "model/ngram_table.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "core/logging.h"
#include "memory/host_task_pool.h"

#if defined(IMP_HAVE_LIBURING)
#include <liburing.h>
#endif

namespace imp {

namespace {

constexpr size_t kAlign = 4096;  // O_DIRECT offset/length/buffer alignment

constexpr uint64_t floor_align(uint64_t v) { return v & ~(uint64_t(kAlign) - 1); }
constexpr uint64_t round_up_k(uint64_t v) { return (v + kAlign - 1) & ~(uint64_t(kAlign) - 1); }

bool pread_exact(int fd, void* dst, size_t len, uint64_t off) {
    size_t got = 0;
    while (got < len) {
        const ssize_t r = pread(fd, static_cast<char*>(dst) + got, len - got, static_cast<off_t>(off + got));
        if (r <= 0)
            return false;
        got += static_cast<size_t>(r);
    }
    return true;
}

// Rows arrive sorted by file offset and deduplicated; two join when they are
// within `window` of the range's first row, so a few KiB of read-ahead replaces a
// syscall. len stays rows*row_bytes for indexing only: a joined range has holes
// between its rows, and the span that covers them is computed in stream_().
std::vector<NGramTable::Range> coalesce(const NGramTable::RowRef* rows, size_t n, size_t row_bytes,
                                        size_t window) {
    std::vector<NGramTable::Range> out;
    size_t i = 0;
    while (i < n) {
        size_t end = i + 1;
        while (end < n && rows[end].off - rows[i].off <= window)
            ++end;
        out.push_back({rows[i].off, i * row_bytes, (end - i) * row_bytes});
        i = end;
    }
    return out;
}

}  // namespace

// Layout and read live together: the row pointers below come from the same spans
// the reader issues, so they cannot disagree.
bool NGramTable::stream_() const {
    if (plan_.empty())
        return true;
    const size_t row_bytes = static_cast<size_t>(head_dim_);
    const std::vector<Range> ranges = coalesce(plan_.data(), plan_.size(), row_bytes, coalesce_bytes_);
    const bool direct = fd_direct_ >= 0;

    std::vector<Span> spans;
    spans.reserve(ranges.size());
    size_t need = 0;
    for (const Range& r : ranges) {
        const size_t first = r.buf_off / row_bytes;
        const size_t rows = r.len / row_bytes;
        // A joined range has holes: its last byte is the last row's end, not
        // first + rows*row_bytes.
        const uint64_t last_end = plan_[first + rows - 1].off + row_bytes;
        Span s{};
        s.file_off = direct ? floor_align(r.file_off) : r.file_off;
        s.file_len = direct ? round_up_k(last_end) - s.file_off : last_end - s.file_off;
        s.buf_off = direct ? static_cast<size_t>(round_up_k(need)) : need;
        s.buf_len = static_cast<size_t>(s.file_len);
        s.first_row = first;
        s.rows = rows;
        if (s.buf_off + s.buf_len > kMaxStaging) {
            IMP_LOG_ERROR("ngram table: staging would need %zu MiB, cap is %zu MiB", (s.buf_off + s.buf_len) >> 20,
                          kMaxStaging >> 20);
            return false;
        }
        need = s.buf_off + s.buf_len;
        spans.push_back(s);
    }

    if (staging_ == nullptr || need > staging_cap_) {
        const size_t cap = std::max<size_t>(need + (need >> 3), size_t(1) << 20);
        void* fresh = nullptr;
        if (posix_memalign(&fresh, kAlign, cap) != 0) {
            IMP_LOG_ERROR("ngram table: staging alloc of %zu MiB failed", cap >> 20);
            return false;
        }
        std::free(staging_);
        staging_ = static_cast<uint8_t*>(fresh);
        staging_cap_ = cap;
    }

    // Each row is addressed by its own offset, not by its position in the span:
    // a joined range's rows are not adjacent in the file.
    row_ptr_.assign(plan_.size(), nullptr);
    for (size_t ri = 0; ri < ranges.size(); ++ri) {
        const Span& s = spans[ri];
        for (size_t k = 0; k < s.rows; ++k)
            row_ptr_[s.first_row + k] = staging_ + s.buf_off + static_cast<size_t>(plan_[s.first_row + k].off -
                                                                                  s.file_off);
    }

    return backend_ == NGramBackend::Uring ? read_uring_(spans) : read_pread_(spans);
}

bool NGramTable::read_pread_(const std::vector<Span>& spans) const {
    std::atomic<uint64_t> direct{0}, buffered{0}, failed{0};
    auto issue = [&](const Span& s) {
        if (fd_direct_ >= 0 && pread_exact(fd_direct_, staging_ + s.buf_off, s.buf_len, s.file_off)) {
            ++direct;
            return;
        }
        if (fd_ >= 0 && pread_exact(fd_, staging_ + s.buf_off, s.buf_len, s.file_off)) {
            ++buffered;
            return;
        }
        ++failed;
    };
    if (pool_ && spans.size() > 1) {
        for (const Span& s : spans)
            pool_->submit([&issue, &s] { issue(s); });
        pool_->wait();
    } else {
        for (const Span& s : spans)
            issue(s);
    }
    std::lock_guard<std::mutex> lk(stats_mu_);
    stats_.ranges += spans.size();
    stats_.bytes_read += static_cast<uint64_t>(plan_.size()) * static_cast<size_t>(head_dim_);
    stats_.direct_reads += direct.load();
    stats_.buffered_reads += buffered.load();
    stats_.read_errors += failed.load();
    return failed.load() == 0;
}

#if defined(IMP_HAVE_LIBURING)

bool NGramTable::read_uring_(const std::vector<Span>& spans) const {
    if (!ring_live_ || queue_depth_ <= 0)
        return read_pread_(spans);
    io_uring* ring = static_cast<io_uring*>(ring_mem_);

    bool ok = true;
    size_t i = 0;
    while (i < spans.size()) {
        const size_t batch = std::min<size_t>(static_cast<size_t>(queue_depth_), spans.size() - i);
        unsigned submitted = 0;
        for (size_t k = 0; k < batch; ++k) {
            const Span& s = spans[i + k];
            io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (sqe == nullptr)
                break;
            io_uring_prep_read(sqe, fd_direct_ >= 0 ? fd_direct_ : fd_, staging_ + s.buf_off,
                               static_cast<unsigned>(s.buf_len), static_cast<off_t>(s.file_off));
            ++submitted;
        }
        if (submitted == 0 || io_uring_submit(ring) < 0) {
            ok = false;
            break;
        }
        for (unsigned c = 0; c < submitted; ++c) {
            io_uring_cqe* cqe = nullptr;
            if (io_uring_wait_cqe(ring, &cqe) < 0 || cqe == nullptr) {
                ok = false;
                break;
            }
            const int res = cqe->res;
            io_uring_cqe_seen(ring, cqe);
            if (res < 0)
                ok = false;
        }
        {
            std::lock_guard<std::mutex> lk(stats_mu_);
            stats_.uring_submits += submitted;
        }
        i += submitted;
        if (!ok)
            break;
    }
    std::lock_guard<std::mutex> lk(stats_mu_);
    stats_.ranges += spans.size();
    stats_.bytes_read += static_cast<uint64_t>(plan_.size()) * static_cast<size_t>(head_dim_);
    if (fd_direct_ >= 0)
        stats_.direct_reads += spans.size();
    else
        stats_.buffered_reads += spans.size();
    if (!ok)
        ++stats_.read_errors;
    return ok;
}

#else  // !IMP_HAVE_LIBURING

bool NGramTable::read_uring_(const std::vector<Span>& spans) const {
    return read_pread_(spans);
}

#endif  // IMP_HAVE_LIBURING

void NGramTable::free_staging_() {
    std::free(staging_);
    staging_ = nullptr;
    staging_cap_ = 0;
}

// A backend that cannot start downgrades to one that can, loudly, rather than
// leaving a gather that fails later.
void NGramTable::init_backend_() {
    if (backend_ != NGramBackend::Uring)
        return;
#if defined(IMP_HAVE_LIBURING)
    // Caller-allocated ring: liburing's io_uring_queue_init fills a struct we own
    // and returns 0, or a negative errno. That is the API every later call
    // (get_sqe, submit, wait_cqe, queue_exit) matches.
    std::free(ring_mem_);
    ring_mem_ = std::calloc(1, sizeof(struct io_uring));
    if (ring_mem_ == nullptr) {
        IMP_LOG_WARN("ngram table: out of memory for an io_uring ring; serving with pread");
        backend_ = NGramBackend::Pread;
        return;
    }
    const int rc = io_uring_queue_init(static_cast<unsigned>(queue_depth_), static_cast<io_uring*>(ring_mem_), 0);
    if (rc < 0) {
        IMP_LOG_WARN("ngram table: io_uring_queue_init(%d) failed (%s); serving with pread", queue_depth_,
                     std::strerror(-rc));
        backend_ = NGramBackend::Pread;
    } else {
        ring_live_ = true;
    }
#else
    IMP_LOG_INFO("ngram table: built without liburing; ple.table_backend=uring serves as pread");
    backend_ = NGramBackend::Pread;
#endif
}

}  // namespace imp
