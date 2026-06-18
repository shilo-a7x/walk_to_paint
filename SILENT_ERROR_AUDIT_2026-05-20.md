# Silent Error Audit (May 20, 2026)

This report captures locations in the repository where exceptions or command failures may be silently ignored, potentially hiding real issues.

## High Severity

1. Failed Optuna trials converted to valid score instead of failing
   - File: `optuna_run.py` (around line 430)
   - Pattern: broad `except Exception` in objective returns `0.0`
   - Risk: runtime/data/model failures look like normal low-scoring trials
   - Recommendation: re-raise unexpected exceptions or explicitly mark trial failed

2. Silent token parse drops in tokenizer
   - File: `src/data/tokenizer.py` (around lines 40, 47, 141, 148)
   - Pattern: `except (IndexError, ValueError): pass`
   - Risk: malformed tokens are silently skipped, creating incomplete maps
   - Recommendation: count malformed tokens and warn once (or fail in strict mode)

3. Silent token parse drops when reconstructing tokenizer from cache
   - File: `src/data/dataset_cache.py` (around lines 131, 138)
   - Pattern: `except (IndexError, ValueError): pass`
   - Risk: cache corruption/format anomalies become invisible
   - Recommendation: warn on malformed cached tokens; optionally strict-fail

4. Silent config mutation failures in outputs path resolver
   - File: `src/utils/paths.py` (around lines 70, 74)
   - Pattern: assignment wrapped in `except Exception: pass`
   - Risk: training may continue with unintended checkpoint/log dirs
   - Recommendation: warn or raise with context

## Medium Severity

5. Broad parse skip in dataset loader
   - File: `src/data/datasets.py` (around line 267)
   - Pattern: `except Exception: continue`
   - Risk: unexpected parse bugs silently drop edges and skew data
   - Recommendation: catch specific parse errors only; report skipped count

6. Bare `except` in top-trial extraction utility
   - File: `scripts/extract_top_trials.py` (around lines 58, 178)
   - Pattern: `except:`
   - Risk: valid studies can be silently skipped
   - Recommendation: catch specific exceptions and print concise reason

7. Silent failures in training/optuna matmul precision setup
   - Files: `run.py` (around line 92), `optuna_run.py` (around line 512)
   - Pattern: `except Exception: pass`
   - Risk: environment/config incompatibilities hidden
   - Recommendation: warn once on failure

8. Silent failure writing trial metadata
   - File: `optuna_run.py` (around line 352)
   - Pattern: `except Exception: pass` around `trial.set_user_attr`
   - Risk: reproducibility metadata disappears silently
   - Recommendation: warn in debug logs

## Low to Medium Severity

9. Silent profiling/plotting failures
   - Files:
     - `src/data/prepare_data.py` (around line 652)
     - `benchmark_pipeline.py` (around line 47)
     - `plot_metrics.py` (around lines 61, 100)
   - Pattern: broad exceptions with `pass`
   - Risk: diagnostics become incomplete without obvious signal
   - Recommendation: keep non-fatal behavior but emit warning/debug log

10. Shell command failures not checked in analysis script
    - File: `analyze_training_results.py` (around lines 20, 25, 30)
    - Pattern: `os.system(...)` with no robust error handling
    - Risk: partial/failed output may appear valid
    - Recommendation: use `subprocess.run(..., check=True)` or verify return code

11. Bare `except` in checkpoint demo parser
    - File: `scripts/eval_checkpoint_demo.py` (around line 41)
    - Pattern: `except:` around parsing expected AUC
    - Risk: malformed checkpoint naming hidden
    - Recommendation: catch `ValueError` and report parse issue

## Notes

- Not every `except ...: pass` is automatically wrong; some are best-effort in exploratory scripts.
- The most critical immediate fix is in `optuna_run.py` where failed trials are converted into a valid numeric objective.

## Suggested Remediation Order

1. `optuna_run.py` objective failure handling
2. `src/data/tokenizer.py` + `src/data/dataset_cache.py` malformed-token visibility
3. `src/utils/paths.py` config assignment failures
4. `src/data/datasets.py` broad parse exception narrowing
5. utility-script hardening (`scripts/extract_top_trials.py`, `analyze_training_results.py`, `scripts/eval_checkpoint_demo.py`)
