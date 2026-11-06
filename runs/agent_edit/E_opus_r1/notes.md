1. Probing where the model goes wrong on order/toggle steps; writing helper scripts via `./py`.

2. Measuring which layers/components push against emitting DONE at goal-reached steps (direct logit attribution).

3. Collecting labeled decision states (expert policy vs model action) across families.

4. Testing whether a "goal reached" signal is linearly readable in the residual stream across families.

5. Plan: closed-form (least-squares, MEMIT-style) edit of the final-layer MLP down_proj, keyed on decision states labelled by an expert policy. First, building the dataset of keys/targets.

6. While that runs, I'll build a retention check (my own 40 QA items + perplexity text) and generic-text keys for the preservation covariance.

7. Still waiting for the state-collection run (it's generating ~1000 labeled agent states); I'll fit the edit once it lands.

8. Data ready (~1200 decision items; many order/toggle errors are at the digit position, e.g. clicking a product again instead of Home). Now fitting the closed-form edit and evaluating.

9. Excellent — 24/24 on fresh seeds with QA answers and perplexity unchanged. Now checking generalization with novel product/user/query names (my own invented pools, not the test pools).

10. Generalizes: 36/36 on novel-name tasks vs. 27/36 baseline. Now shrinking the edit: restrict it to a small set of neurons (columns of down_proj) and refit.

11. Sparse-column fits leave larger residuals. Let me build a fast offline scorer (exact logits from stored residuals) to compare row/column-restricted variants before running episodes.

12. Restricting to output rows works well: 128 rows × all neurons (1.15M values, 0.07% of weights) keeps 96% target accuracy. Evaluating it end-to-end.

13. 64-row edit: 36/36, retention unchanged. Now saving it as `edited/` (checking dtype first) and running the official harness.

14. Corpus/covariance build is running in the background (~60k tokens). Meanwhile writing the refit and out-of-sample evaluation scripts.

15. Rebuilding the generic-text covariance (fixed an MPS float64 bug that killed the first attempt).

16. Covariance built from 52k generic tokens. Refitting with it.

17. Refit grid is running on CPU; now running episodes for the 256-row, well-preserved variant on novel-name tasks.

18. 32/36 (baseline 27/36): better preservation costs some fix strength. Checking the weaker-preservation grid.