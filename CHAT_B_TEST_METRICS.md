# CHAT B: Test Metrics Tracking

## Task B1: Per-Epoch Test Metrics During Training

### Current Problem
Currently, test metrics are only evaluated at the end of training (after early stopping triggers or max epochs reached). This means:
- We can't see test AUC trends during training
- We can't debug overfitting (compare test vs val curves per epoch)
- TensorBoard only shows val metrics, not test
- Test metrics are a surprise at the end, not monitored

### Your Task
1. **Add test metrics logging per epoch**:
   - In `src/model/lit_model.py`, modify `on_validation_epoch_end`
   - Add logic to also run test_dataloader after val_dataloader
   - Log: `test_auc_epoch`, `test_f1_epoch`, `test_acc_epoch` to TensorBoard
   - Use same metrics_manager as val to maintain consistency

2. **Ensure early stopping still works**:
   - Early stopping monitors `val_auc_epoch` (don't change)
   - Test metrics are logged but NOT used for early stopping
   - Test logging should not slow down training significantly

3. **Update training config** (config.yaml):
   - Add optional flag `training.eval_test_per_epoch: true/false` (default false)
   - Allows users to disable test logging if they want faster training
   - Document in CONFIG_GUIDE.md

4. **Verify with existing results**:
   - Run training on epinions for 3 epochs with test logging enabled
   - Check TensorBoard logs have test_auc, test_f1, test_acc curves
   - Verify it doesn't break existing checkpoints or early stopping

### Files to Modify
- `src/model/lit_model.py` (add test metrics per epoch)
- `src/training/train.py` (may need to set test_dataloader in Trainer)
- `config.yaml` (add eval_test_per_epoch flag)

### Success Criteria
✅ Test metrics logged to TensorBoard after every validation epoch  
✅ Early stopping only monitors val (unchanged behavior)  
✅ Test curves appear in TensorBoard alongside val curves  
✅ Training speed not significantly impacted  
✅ Works with existing checkpoints + eval_only mode  

### Output
When complete, provide:
- Screenshot of TensorBoard showing test_auc vs val_auc curves
- Summary: "Test metrics tracking enabled for [X] configurations"
- Any config changes needed for other chats

---

## Notes
- **Depends on**: Chat A (config system) - make sure you wait for A1 to complete
- **Doesn't conflict with**: Any other chat (pure logging addition)
- **Integration point**: Chat F will use these metrics for dashboards/comparison
