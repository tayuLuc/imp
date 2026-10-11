#pragma once

// One of ten RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other nine's churn.
// dispatch_policy.h still includes all ten.
//
// Qwen4Exp PLE (per-layer n-gram embedding table): F8 shards, too large to be
// resident, so every backend reads only the rows a gather selects.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct PLE {
    // How the n-gram table's rows are read:
    //   "mmap"   (default) shard file mapped once, MADV_RANDOM, MADV_WILLNEED
    //             per row; the page cache owns residency.
    //   "pread"  rows are read with pread into host staging (O_DIRECT when the
    //             filesystem allows it); page cache is not on the read path.
    //   "uring"  pread, submitted as a bounded io_uring batch; falls back to
    //             "pread" when the build has no liburing or the kernel refuses.
    std::string table_backend = "mmap";
    // Host threads issuing the reads for the streaming backends.
    // 0 = auto (4, capped by the online cores).
    int io_threads = 0;
    // Rows whose file offsets fall within this window of each other are read as
    // one range. 0 = auto (64 KiB). Bigger = fewer syscalls, more bytes moved.
    int coalesce_kib = 0;
    // io_uring SQEs in flight per gather. 0 = auto (min(64, ranges)).
    int queue_depth = 0;
    // Log the backend actually used and the row/range counts once per process.
    bool log_stats = true;
};

}  // namespace imp::cfg
