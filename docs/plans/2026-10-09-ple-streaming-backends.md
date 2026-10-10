# PLE streaming backends: measurement plan

Status: implementation merged in the fork; numbers pending a GPU run.

## What is being compared

`ple.table_backend` selects who faults the PLE n-gram table in:

- `mmap` (default, unchanged): the shard file is mapped once, `MADV_RANDOM`,
  `MADV_WILLNEED` per row. The page cache owns residency.
- `pread`: the rows a gather selects are read into a 4 KiB-aligned host staging
  buffer with `pread`, O_DIRECT where the filesystem accepts it, on
  `ple.io_threads` workers.
- `uring`: the same reads submitted as a bounded io_uring batch
  (`ple.queue_depth` SQEs in flight). Without liburing it logs once and serves
  as `pread`.

The three return byte-identical rows; the question is what they cost on a box
where the table does not fit in RAM.

## Why this box

Its PLE is 47.7 GiB (FP8) against 59 GiB of host RAM, and the host expert tier
holds 56.25 GiB of NVFP4 experts at `--memory=64g`. mmap + page cache is the
configuration where reclaim decides latency; that is the case worth measuring.

## Run

Same box, same process, one run per backend over the same prompt set, the
backend varied by `--set`. `$MODEL` is a checkpoint directory: whatever path the
host mounts as the model root, the point is that it is not on a filesystem-sized
tmpfs. A container mount (`-v <host>:/models`) is the usual shape:

```bash
MODEL=/models/qwen3.8-flash-next-nvfp4
for backend in mmap pread uring; do
  imp-cli --model "$MODEL" \
          --set "ple.table_backend=$backend" \
          --bench-pp 512 --bench 128
done
```

Record per backend:

1. prefill tok/s and decode tok/s (median of 3, single session per backend as
   the perf gate requires).
2. host RSS over the run, sampled (`VmRSS` from /proc): this is the number the
   change exists to move.
3. page cache share of the shard file before and after
   (`grep -E 'Cached|MemAvailable' /proc/meminfo`).
4. `stats()` counters the table logs at unload: rows, ranges, direct vs buffered
   reads, io_uring submits.

## Falsification

The feature claims the page cache stops absorbing the table. If `pread` and
`uring` show host RSS within noise of `mmap`, the staging path is not the
binding constraint and the claim is refuted regardless of throughput.

## Open items

1. `ple.coalesce_kib` sweep (0 = auto 64 KiB): the decode working set is about
   17.6 rows spread over 47.7 GiB, so a wider window mostly costs read-ahead
   bytes. Measure before raising it.
2. `ple.io_threads` sweep: 4 by default, not derived from the device.
3. Multi-file PLE tables are still refused by the loader, so a model that ships
   its table as many shards cannot be measured until that lands.
