#!/bin/bash

# Script to clean up old/redundant MD files from repository
# Run with: bash cleanup_old_docs.sh

cd /home/dsi/shilo_avital/yolo_lab/walk_to_paint

# Files to delete (old, redundant, outdated)
DELETE_FILES=(
    "CHAT_A_CONFIG_REPRODUCIBILITY.md"
    "CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md"
    "CHAT_B_TEST_METRICS.md"
    "CHAT_C_PREDICTION_CACHING.md"
    "CHAT_D_DATA_PIPELINE.md"
    "CHAT_D_DATA_PIPELINE_UPDATED.md"
    "CHAT_E_SEED_CLEANUP.md"
    "CHAT_E_SEED_CLEANUP_UPDATED.md"
    "CHAT_F_AGGREGATOR_INTEGRATION.md"
    "COMEBACK_SUMMARY.md"
    "CHAT_HISTORY_SUMMARY.md"
    "PROJECT_INDEX.md"
    "COMPLETE_ANALYSIS.md"
    "FINAL_SUMMARY.md"
    "PROMPT_UPDATES_SUMMARY.md"
    "QUICK_REFERENCE.md"
    "FILES_SUMMARY.md"
    "IMPLEMENTATION_CHECKLIST.md"
    "GET_BACK_TO_WORK_PLAN.md"
    "INDEX.md"
    "VISUAL_SUMMARY.md"
    "ARCHITECTURE.md"
    "WIKI_RFA_EVALUATION.md"
    "WIKI_RFA_EVAL_REPORT.md"
    "CHECKPOINT_LOADING_FIX.md"
    "DATA_PIPELINE_ANALYSIS.md"
    "DATA_OPTIMIZATION_ANALYSIS.md"
    "A1_IMPACT_ANALYSIS.md"
    "TASK_A1_REPRODUCIBILITY_FIXES.md"
    "TASK_A1_IMPLEMENTATION_SUMMARY.md"
    "TASK_A1_FINAL_STATUS.md"
    "AGGREGATOR_CONFIG.md"
    "OPTIMIZATION_GUIDE.md"
    "EDGE_AGGREGATION_GUIDE.md"
    "REPRODUCIBILITY_REVIEW.md"
    "TASK_A6_PROMPT.md"
    "TASK_A6_SPLIT_SEMANTICS.md"
    "TASK_A6_SPLIT_SEMANTICS_EXPLAINED.md"
    "TASK_A6_VERIFICATION_CHECKLIST.md"
    "TASK_A6_STRATIFIED_SPLITTING_IMPLEMENTATION.md"
    "TASK_A6_COMPLETE.md"
    "TASK_A6_IMPLEMENTATION_SUMMARY.md"
    "FULL_PROJECT_STATUS.md"
)

echo "🗑️  Cleaning up old documentation files..."
echo "Files to delete: ${#DELETE_FILES[@]}"
echo ""

for file in "${DELETE_FILES[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "✓ Deleted: $file"
    else
        echo "⊘ Not found: $file"
    fi
done

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📋 Files kept:"
echo "  - README.md"
echo "  - WALK_REPRODUCIBILITY_EXPLAINED.md"
echo "  - WALK_SOLUTION_COMPLETE.md"
echo "  - CONFIG_GUIDE.md"
echo "  - COMPREHENSIVE_RETHINKING.md (new)"
echo ""
