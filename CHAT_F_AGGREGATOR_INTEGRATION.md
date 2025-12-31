# CHAT F: Aggregator Integration Strategy

## Task F1: Design & Implement Aggregator Integration Architecture

### Current Problem
The aggregator currently exists as a separate pipeline:
1. Train transformer (outputs checkpoint)
2. Extract edge scores from transformer
3. Train aggregator on top of features

This separation is fine for research, but has limitations:
- Aggregator can't benefit from transformer fine-tuning
- Pipeline is brittle if transformer retrains
- No joint optimization
- Hard to deploy (two-stage pipeline)

### Your Task
1. **Design three integration strategies**:

   **Strategy A: Separate Module** (current state)
   - Keep transformer and aggregator independent
   - Pros: Simple, modular, easy to experiment
   - Cons: No joint learning, data flow is manual
   - Deployment: Load transformer → extract → load aggregator (2 models)

   **Strategy B: Joint Training** (transformer + aggregator end-to-end)
   - Fine-tune transformer while aggregator trains
   - Pros: Joint optimization, better final AUC
   - Cons: Complex training loop, longer training time
   - Deployment: Single model with both heads (1 model)

   **Strategy C: Pipeline Integration** (aggregator as inference component)
   - Transformer trained normally
   - Aggregator trained after, stored separately
   - At inference: transformer outputs → aggregator refines (2 stages)
   - Pros: Separate concerns, reusable components
   - Cons: Inference latency (two forward passes per edge)

2. **Evaluate each strategy**:
   - Implementation complexity (1-10 scale)
   - Training time increase (% overhead)
   - Inference latency (ms per edge)
   - Experimentation flexibility (1-10 scale)
   - Deployment complexity (1-10 scale)
   - Expected AUC improvement vs baseline
   - Create comparison table in `docs/AGGREGATOR_STRATEGY_COMPARISON.md`

3. **Implement chosen strategy**:
   - You'll decide which strategy to implement based on evaluation
   - Likely: Strategy C (pipeline integration) as good balance of simplicity + improvement
   - Or: Strategy A (separate) if no time for implementation

4. **Update pipeline code**:
   - If Strategy A → create `scripts/run_transformer_and_aggregator.py` (orchestrator)
   - If Strategy C → update eval to load both models and run 2-stage inference
   - Add configuration options for strategy choice to `config.yaml`

5. **Create unified evaluation**:
   - Script: `scripts/evaluate_full_pipeline.py`
   - Takes dataset name, transformer checkpoint, aggregator checkpoint
   - Outputs: transformer AUC, aggregator AUC, improvement %
   - Creates comparison visualizations (curves, confusion matrices)

6. **Add TensorBoard dashboards**:
   - Create custom dashboards comparing all strategies
   - Show: transformer AUC, aggregator AUC, improvement per dataset
   - Add position-based insights from Chat C (if available)
   - Output: `tensorboard_dashboards/aggregator_comparison.json`

7. **Document architecture decision**:
   - Create `docs/AGGREGATOR_INTEGRATION.md`:
     - Explains chosen strategy
     - Architecture diagram (ASCII or actual diagram)
     - When to use each strategy (guidelines for future)
     - Integration points with main training loop
     - Deployment instructions

### Files to Create/Modify
- Create `scripts/evaluate_full_pipeline.py` (new)
- Create `docs/AGGREGATOR_STRATEGY_COMPARISON.md` (analysis)
- Create `docs/AGGREGATOR_INTEGRATION.md` (final design)
- Possibly modify `config.yaml` (add aggregator config options)
- Possibly modify `scripts/train_aggregator.py` (if joint training)

### Success Criteria
✅ All 3 strategies evaluated (pros/cons documented)  
✅ Strategy comparison table created  
✅ Chosen strategy implemented and tested  
✅ evaluate_full_pipeline.py works for all datasets  
✅ Performance regression testing script available  
✅ Architecture documented with diagrams  
✅ Deployment guide written  

### Example: Strategy Comparison Table
```
| Factor              | Separate | Joint | Pipeline |
|---------------------|----------|-------|----------|
| Implementation      |    2     |   8   |    5     |
| Training overhead   |   0%     |  40%  |   10%    |
| Inference latency   |   2ms    |  1ms  |   3ms    |
| Experimentation     |    9     |   5   |    8     |
| AUC improvement     |  +2.3%   | +3.0% |  +2.3%   |
| Recommendation      |   EASY   | BEST  |   GOOD   |
```

---

## Timeline & Dependencies
- **Depends on**: Chat A (config), B (metrics), C (prediction caching optional), D (pipeline)
- **Can start**: After others have produced results and documented findings
- **Estimated time**: 2-3 hours (design + evaluation + implementation + documentation)

## Integration with Other Chats
- **From Chat B**: Use test metrics for comparison (transformer vs aggregator)
- **From Chat C**: Use position-based insights (if available) to explain improvements
- **From Chat D**: Use optimized data pipeline for evaluation
- **Input**: Results from all previous chats

## Deliverables Summary
1. Strategy comparison document with pros/cons
2. Implemented integration (at least Strategy A, ideally C)
3. evaluate_full_pipeline.py script (reusable)
4. Architecture documentation with decisions explained
5. Performance regression test suite

---

## Example Architecture Diagram (Strategy C)
```
Training Phase:
  Edge List → Transformer Training → Best Checkpoint
             → Extract Features → Aggregator Training → Best Agg Checkpoint

Inference Phase:
  Test Edge
    ↓
  [Transformer] → Edge Probabilities
    ↓
  Feature Extraction (percentiles, stats)
    ↓
  [Aggregator] → Final Prediction (with confidence)
    ↓
  Result
```

---

## Final Note
After this chat completes, the project will have:
✅ Unified reproducible configuration (A)
✅ Validated data pipeline (A6)
✅ Per-epoch test metrics (B)
✅ Cached raw predictions + analysis (C)
✅ Optimized data I/O (D)
✅ Clean, deterministic seed handling (E)
✅ Clear aggregator integration strategy (F)

This positions the project for:
- Easy experimentation (caching, reproducibility)
- Performance monitoring (test metrics per epoch)
- Further optimization (data pipeline tuning)
- Production deployment (clear architecture)
