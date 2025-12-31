# CHAT D: Data Pipeline Review & Optimization

## Task D1: Walk Sampling Algorithm Analysis & Optimization Review

### Current Problem
We need to understand and potentially optimize the walk sampling pipeline:
- `src/data/walk_sampler.py` uses multiprocessing.Pool to sample random walks
- No documentation of algorithm, complexity, or optimization opportunities
- File I/O and multiprocessing overhead unknown
- Unclear if current approach is efficient for large graphs

### Your Task
1. **Understand the walk sampling algorithm**:
   - Read `src/data/walk_sampler.py` carefully
   - Document: How are random walks sampled? (biased/unbiased? BFS/DFS/random restart?)
   - What's the complexity? (O(n*walks*walk_length)?
   - What are the assumptions? (node range [0, n-1]? no multiedges?)

2. **Map the multiprocessing strategy**:
   - How many workers? (from config or hardcoded?)
   - What's the bottleneck? (CPU bound or I/O bound?)
   - How much memory per worker?

3. **Identify optimization opportunities**:
   - Is there redundant computation? (e.g., same walks sampled multiple times)
   - Caching strategy: where are walks stored? How often accessed?
   - Could we vectorize instead of multiprocessing?
   - What would be the impact of each optimization?

4. **Benchmark current approach**:
   - Create `scripts/benchmark_walks.py` that:
     - Times walk sampling for each dataset
     - Measures memory usage (RSS)
     - Reports throughput (walks/second)
     - Compares with simple single-threaded baseline
   - Output to `benchmark_results/walk_sampling_{dataset}.csv`

5. **Write algorithm documentation**:
   - Create `docs/WALK_SAMPLING.md`:
     - Algorithm description (with citations if applicable)
     - Complexity analysis
     - Multiprocessing rationale
     - Optimization opportunities (with trade-offs)
     - Benchmark results

### Files to Review/Modify
- `src/data/walk_sampler.py` (read carefully, no changes yet)
- `src/data/prepare_data.py` (understand how walks are cached/loaded)
- Create `scripts/benchmark_walks.py`
- Create `docs/WALK_SAMPLING.md`

### Success Criteria
✅ Algorithm documented in WALK_SAMPLING.md  
✅ Complexity analysis provided  
✅ Benchmarks show current performance (time + memory)  
✅ Top 3 optimization opportunities identified with pros/cons  
✅ Recommendations for D2  

### Example Output (WALK_SAMPLING.md)
```
## Algorithm
Random walk sampling using numpy.random.choice with replacement.
Samples `num_walks` walks per edge from Graph neighbors.

## Complexity
O(E * num_walks * walk_length) where E = number of edges

## Bottlenecks
- Identified: I/O bound (disk cache write/read)
- Solution: Use in-memory walk storage with LRU cache
- Speedup estimate: 2-3x for repeated epochs
```

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
- **D1 is independent**: Can start immediately (pure analysis, no code changes)
- **D2 depends on D1**: Wait for D1 findings before optimizing
- **Estimated time**: D1 (2-3h analysis), D2 (3-4h implementation + testing)

## Output Artifacts
- `scripts/benchmark_walks.py` (reusable benchmark tool)
- `docs/WALK_SAMPLING.md` (algorithm documentation)
- `docs/DATA_PIPELINE.md` (after D2, I/O guide)
- `benchmark_results/` directory (benchmark CSVs + plots)

## Notes
- **Low risk**: D1 is pure documentation, no code changes
- **Medium risk**: D2 changes I/O paths, requires validation
- **Validation**: Run full training on 1 dataset (epinions fastest) after D2
- **Rollback**: Keep old code in git if optimization causes issues
