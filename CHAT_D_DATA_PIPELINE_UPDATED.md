# CHAT D: Data Pipeline Review & Optimization

## Task D1: Walk Sampling Algorithm Verification & Optimization Review

### Status Update: Walk Reproducibility COMPLETE ✅
Task A1 has already completed comprehensive analysis of the walk sampling algorithm. The following resources document the solution:
- [WALK_REPRODUCIBILITY_EXPLAINED.md](WALK_REPRODUCIBILITY_EXPLAINED.md) - Complete algorithm explanation with guarantees
- [TASK_A1_FINAL_STATUS.md](TASK_A1_FINAL_STATUS.md#5--walk-sampling-reproducibility) - Implementation details and test results

**Key finding from A1**: Walk reproducibility is **GUARANTEED** via:
1. Per-walk deterministic seeding: `walk[i]` uses seed `base_seed + i`
2. Task sorting by task_id before concatenation (multiprocessing order doesn't matter)
3. Result: Bit-for-bit identical walk files regardless of worker count (1, 2, 4, 8+)

### Current Problem
Your task is now to verify this solution and identify optimization opportunities. We also need to understand the walk sampling algorithm and look for potential performance improvements.

### Your Task
1. **Verify A1's walk reproducibility solution**:
   - Read [WALK_REPRODUCIBILITY_EXPLAINED.md](WALK_REPRODUCIBILITY_EXPLAINED.md) completely
   - Understand the per-walk seeding mechanism (why walk[i] uses seed base+i)
   - Understand why sorting by task_id guarantees correct order
   - Run the walk reproducibility test from A1 with different worker counts
   - Verify output is bit-for-bit identical

2. **Understand the walk sampling algorithm**:
   - Read `src/data/walk_sampler.py` carefully
   - Document: How are random walks sampled? (unbiased? with specific distribution?)
   - What's the complexity? (O(n*walks*walk_length)?)
   - What are the assumptions? (nodes [0, n-1]? no multiedges?)
   - How does the starting node selection work (per-walk RNG seeding)?

3. **Map the multiprocessing strategy**:
   - How many workers? (from config `preprocess.num_workers` which defaults to 8)
   - What's the bottleneck? (CPU bound or I/O bound?)
   - How much memory per worker?
   - How are tasks distributed (chunks of consecutive walks)?

4. **Identify optimization opportunities**:
   - Is there redundant computation?
   - Could vectorization help instead of multiprocessing?
   - Current walk caching strategy: where stored, how accessed?
   - What would be the impact of each optimization? (estimate time savings)

5. **Benchmark current approach**:
   - Create `scripts/benchmark_walks.py` that:
     - Times walk sampling for each dataset
     - Measures memory usage (RSS)
     - Reports throughput (walks/second)
     - Compares with simple single-threaded baseline
   - Output to `benchmark_results/walk_sampling_{dataset}.csv`

6. **Write algorithm documentation**:
   - Update or create `docs/WALK_SAMPLING.md`:
     - Algorithm description (reference A1's work, don't duplicate)
     - Complexity analysis
     - Multiprocessing rationale
     - Reproducibility guarantee explanation
     - Top 3 optimization opportunities (with trade-offs)
     - Benchmark results

### Files to Review/Modify
- `src/data/walk_sampler.py` (read carefully)
- `src/data/prepare_data.py` (understand walk integration)
- Create `scripts/benchmark_walks.py`
- Create/Update `docs/WALK_SAMPLING.md`

### Success Criteria
✅ A1's walk reproducibility solution verified  
✅ Algorithm documented in WALK_SAMPLING.md  
✅ Complexity analysis provided  
✅ Benchmarks show current performance (time + memory) for all datasets  
✅ Top 3 optimization opportunities identified with pros/cons  
✅ Recommendations clear for D2  

---

## Task D2: Data Pipeline File Format & I/O Optimization

### Current Problem (after D1)
Based on D1 findings, implement optimizations for:
- File format (JSON vs pickle vs parquet vs HDF5)
- Caching strategy (where/how to store intermediate data)
- I/O patterns (sequential, random, streaming)

### Your Task (do this AFTER D1)
1. **Benchmark file formats**:
   - Current: JSON caches, torch.save for walks
   - Test: pickle, parquet, HDF5 for same data
   - Metrics: read speed, write speed, file size, memory footprint
   - Output to `benchmark_results/format_comparison.csv`

2. **Profile I/O patterns**:
   - Use `py-spy` or built-in profiler to see time spent in I/O
   - Identify hotspots: is it JSON loading? torch.save? dataset access?
   - Measure cache hit rates

3. **Implement recommended optimizations**:
   - Switch to faster format (if D1 identified this)
   - Optimize cache invalidation strategy
   - Add memory-mapped option for large walks
   - Implement lazy loading where applicable

4. **Validate improvements**:
   - Run full training with new pipeline
   - Compare: wall-clock time, peak memory, convergence behavior
   - Ensure reproducibility not affected

5. **Update documentation**:
   - Add to `docs/DATA_PIPELINE.md`:
     - File format rationale
     - I/O performance metrics
     - Tuning recommendations

### Files to Modify (depends on D1 findings)
- Possibly: `src/data/prepare_data.py` (format/caching logic)
- Possibly: `src/data/walk_sampler.py` (walk storage)
- Benchmark scripts (reuse from D1)

### Success Criteria
✅ File format optimized (either confirm current is best, or migrate)  
✅ I/O time reduced by >10% compared to baseline (goal)  
✅ Reproducibility unaffected  
✅ Pipeline documentation updated  
✅ Performance report generated  

---

## Timeline & Dependencies
- **D1 is mostly verification**: Uses A1's work; focuses on benchmarking and optimization identification
- **D2 depends on D1**: Wait for D1 findings before optimizing
- **Estimated time**: D1 (2-3h verification + benchmarking), D2 (3-4h implementation + testing)

## Output Artifacts
- `scripts/benchmark_walks.py` (reusable benchmark tool)
- `docs/WALK_SAMPLING.md` (algorithm documentation with A1 references)
- `docs/DATA_PIPELINE.md` (after D2, I/O guide)
- `benchmark_results/` directory (benchmark CSVs + plots)

## Notes
- **D1 is now verification + optimization search**: A1 already solved reproducibility
- **Low implementation risk**: D1 is mostly benchmarking and analysis
- **Medium risk**: D2 changes I/O paths, requires validation
- **Validation**: Run full training on 1 dataset (epinions fastest) after D2
- **Rollback**: Keep old code in git if optimization causes issues
