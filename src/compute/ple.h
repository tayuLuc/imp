#pragma once
// Qwen4Exp PLE (n-gram per-layer embedding) kernels. FP16 in/out, FP32 math. The projections
// and grouped norms around them reuse gemm() and hc_grouped_rmsnorm(); executor_ple.cpp wires it.

#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// In place over q: gv[t, s*d + j] = sigmoid(g) * value[t, j] with
// g = dot(key[t, s, :], q[t, s, :]) / sqrt(d), g = sqrt(max(|g|, 1e-6)) * sign(g).
void ple_gate_value(const Tensor& key, Tensor& q_gv, const Tensor& value, int hc, int d, cudaStream_t stream);

// hidden[t, c] += gv[t, c] + silu(sum_k w[c, k] * x[t - (kernel-1-k)*dilation, c]) with
// x = conv_state (state_len = (kernel-1)*dilation rows, the sequence's past) ++ gvn; then
// conv_state <- the last state_len rows of x. w is depthwise [channels, kernel].
// n_seq sequences of n/n_seq rows each; sequence s's [state_len, channels] FP16 rows start at
// conv_state + (slots ? slots[s] : 0) * slot_stride halves (slots: DEVICE, graph-stable).
// Single sequence: d_real_n (device) caps the committed rows (pad rows), snap_state gets the state
// after d_snap_n rows (verify row-0 snapshot). All three nullable.
void ple_conv_add(const Tensor& gv, const Tensor& gvn, const Tensor& w, void* conv_state, int64_t slot_stride,
                  const int* slots, int n_seq, Tensor& hidden, int channels, int kernel, int dilation,
                  cudaStream_t stream, void* snap_state = nullptr, const int* d_snap_n = nullptr,
                  const int* d_real_n = nullptr);

// Device-side n-gram gather. Reads the mapped shard file directly, so it can be captured in
// a CUDA graph where a host gather cannot. Every pointer is a device pointer except
// map_base, which is a host mmap the GPU dereferences through HMM. Byte-identical to
// NGramTable::gather, which stays the reference implementation.
//
// count rows of head_dim halves into d_out, one id per row. Shard geometry comes from the
// flat arrays NGramTable publishes. The caller must have confirmed the table is mapped.
void ple_gather(const int64_t* d_ids, uint32_t count, const uint8_t* map_base,
                const int64_t* d_shard_row_starts, const uint64_t* d_shard_byte_offs, int n_shards,
                int64_t total_rows, int head_dim, float scale, uint16_t* d_out, cudaStream_t stream);

// Grid sizing for ple_gather, kept here so it is checkable without a GPU.
//
// A CUDA graph bakes the launch geometry into the captured node, so a grid that changes between
// steps needs a separate capture per step. count is n * n_heads, and under continuous batching n
// moves with the batch composition, so launching exactly `count` blocks would re-capture constantly.
// Launching a fixed bucket instead, and letting the kernel's own `row >= count` guard drop the
// tail, keeps the geometry constant while wasting at most one bucket of idle blocks.
//
// 4096 is a compromise, not a tuned number: a decode step asks for n*n_heads rows, which is small,
// while a prefill chunk can ask for thousands. The bucket has to cover the prefill case cheaply,
// and idle blocks exit on their first instruction.
inline constexpr uint32_t ple_gather_grid_bucket() { return 4096; }

// Blocks to launch for `count` rows: a whole number of buckets, never fewer than count.
inline constexpr uint32_t ple_gather_blocks(uint32_t count) {
    return count == 0 ? 0u
                      : ((count + ple_gather_grid_bucket() - 1) / ple_gather_grid_bucket()) *
                            ple_gather_grid_bucket();
}

static_assert(ple_gather_blocks(0) == 0, "an empty gather launches nothing");
static_assert(ple_gather_blocks(1) == ple_gather_grid_bucket(), "one row still fills a bucket");
static_assert(ple_gather_blocks(ple_gather_grid_bucket()) == ple_gather_grid_bucket(),
              "an exact bucket is not padded");
static_assert(ple_gather_blocks(ple_gather_grid_bucket() + 1) == ple_gather_grid_bucket() * 2,
              "one row past a bucket needs a second one");
static_assert(ple_gather_blocks(100000) % ple_gather_grid_bucket() == 0,
              "the grid is always a whole number of buckets");
static_assert(ple_gather_blocks(320001536u) >= 320001536u, "the grid always covers the rows asked for");

}  // namespace imp
