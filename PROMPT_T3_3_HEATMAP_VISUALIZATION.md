# Prompt: T3.3 Triplet Heatmap Visualization (Proof of Concept)

## Goal

Create the professor’s requested visualization:
> “Produce triplets (distance from start, distance from end, 0/1 correct) then plot the average of the 0/1 flag as a function of the two first and plot a heatmap.”

## Input

Triplets from T3.2:

- `outputs/aggregation/<dataset>/strategy_mean/triplets_<split>.pkl`

## Heatmap Definition

- X-axis: distance from start (raw edge count)
- Y-axis: distance from end (raw edge count)
- Color: average of `correct` for all triplets with the same (x, y)

## Implementation (POC)

1. Load triplets
2. Use raw integer distances; group by exact (dist_from_start, dist_from_end)
3. For each (x, y):
   - avg_correct = mean(correct)
4. Create a 2D grid sized by max distances + 1
5. Plot heatmap:
   - `imshow` or `pcolormesh`
   - Colorbar: “Avg Correct”
   - Labels: “Distance from Start (edges)”, “Distance from End (edges)”

## Output

- `outputs/aggregation/<dataset>/strategy_mean/heatmap_<split>.png`

## Success Criteria

- ✅ Heatmap saved to disk
- ✅ Axes labeled correctly
- ✅ Values are in [0,1]
- ✅ Works on one dataset + split

## Notes

- Keep minimal; no refactor.
- Use `matplotlib` + `numpy` only.
