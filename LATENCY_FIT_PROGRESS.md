# Latency Fitting Progress

## Goal

This document records the work done so far on fitting DistServe latency models, starting from the original SLO-scale analysis and continuing through the fresh single-forward reruns and formula search.

Primary target:

- average relative error around `2%`
- maximum relative error at most `5%`

Current focus:

- `llama_1B`
- `llama_3B`
- `llama_8B`
- `llama_7B` was investigated earlier, but later deprioritized because of poor performance and a likely model replacement

## Earlier Work

### 1. SLO-scale sweep support

Added SLO-scale sweep and plotting support:

- `simdistserve/benchmarks/merged_analyze.py`
- `simdistserve/benchmarks/plot_slo_scale_for_rate.py`
- `simdistserve/tests/test_slo_scale_sweep.py`

Generated results under:

- `simdistserve/benchmarks/results/slo_scale_plots/`

### 2. Model coverage and 7B check

Validated the original sweep on:

- `llama_1B`
- `llama_3B`
- `llama_8B`

Also checked `llama_7B`, but its fit quality was poor. Later experiments stopped prioritizing 7B after the decision to switch models.

### 3. `sum_remaining_output_tokens` ablation

Tested the effect of removing only the `sum_remaining_output_tokens` factor while leaving the other decode factors in place.

Relevant output:

- `/users/rh/tmp/distserve_no_remaining_ablation/analysis/comparison_summary.json`

Conclusion:

- removing `sum_remaining_output_tokens` did not clearly improve the final SLO-curve fitting quality
- the effect was mostly neutral overall

### 4. Formula interpretation versus DistServe paper

Read the DistServe paper appendix and compared the paper formulas to the repo formulas.

Paper-aligned interpretation:

- prefill is driven by prompt-length terms like `sum(l_i)` and `sum(l_i^2)`
- decode is driven by current context-length terms
- the paper does not include a remaining-token term

Important repo observation:

- the repo’s live decode fit had an empirical `sum_remaining_output_tokens` feature that is not from the paper

### 5. Exhaustive formula search on existing serving benchmark traces

Built temporary scripts to search over mathematically motivated feature combinations using the existing benchmark traces:

- `/users/rh/tmp/search_latency_formulas.py`
- `/users/rh/tmp/search_latency_formulas_exhaustive.py`

Key outputs:

- `/users/rh/tmp/latency_formula_search/distserve_best_summary.json`
- `/users/rh/tmp/latency_formula_search/distserve_exhaustive_summary.json`

Conclusion from that stage:

- better feature engineering alone did not recover the target on the existing trace-derived data
- this suggested either data quality issues or target-construction issues

## Shift To Fresh Single-Forward Data

Because the user suspected overfitting and data-quality issues, the work moved to fresh single-forward profiling under `evaluation/0-test-single-forward-performance`.

### Constraints handled

- Ray is used locally, per `utils/ray_guildline.md`
- other users may use the GPUs
- destructive cleanup such as `pkill -f ray` was intentionally avoided

### Safe execution approach

Instead of touching the shared Ray head aggressively, a tmp-only local Ray runtime was used with explicit GPU selection through `CUDA_VISIBLE_DEVICES`.

Tmp-only profiler and analysis scripts were added:

- `/users/rh/tmp/fresh_distserve_formula_fit/profile_distserve_identical_batches.py`
- `/users/rh/tmp/fresh_distserve_formula_fit/analyze_distserve_identical_batches.py`
- `/users/rh/tmp/fresh_distserve_formula_fit/analyze_single_forward_db_and_fresh.py`
- `/users/rh/tmp/fresh_distserve_formula_fit/analyze_corrected_aggregation.py`

## Important Discovery: Target Construction Was Partly Wrong

Two major issues were identified in the fresh-data analysis.

### Prefill aggregation issue

Prefill time should not depend on `output_len`.

The correct prefill target is better represented by aggregating samples by:

- `(batch_size, input_len)`

instead of:

- `(batch_size, input_len, output_len)`

### Decode aggregation issue when removing remaining-token term

If decode does not include a remaining-token feature, then decode should be modeled using:

- `(batch_size, current_context_len)`

instead of the full request tuple.

Otherwise the target contains extra variance from unrelated request grouping.

This correction was one of the biggest improvements in the whole project.

## Fresh Reruns

### Initial fresh reruns

Fresh reruns were collected under:

- `/users/rh/tmp/fresh_distserve_formula_fit/raw`

Findings:

- `llama_1B` was extremely noisy with only 2 measurement rounds
- `llama_3B` and `llama_8B` decode looked much better once aggregated correctly
- DistServe/SwiftTransformer repeatedly hit a CUDA illegal-memory-access bug on the very last `llama_3B` config

To prevent losing results, the profiler was updated to checkpoint after every completed config.

### Stable reruns

More stable data was then collected with:

- `warmup_rounds = 2`
- `measure_rounds = 5`

Outputs:

- `/users/rh/tmp/fresh_distserve_formula_fit/raw_stable`

Corrected-aggregation analysis:

- `/users/rh/tmp/fresh_distserve_formula_fit/corrected_aggregation_analysis_stable/corrected_aggregation_summary.json`

## Best Current Decode Result

Using corrected aggregation and a simple mathematically motivated decode model:

- features: `const + bs_ctx + ctx_blocks + bs_ctx_blocks`

stable results are:

### `llama_3B`

- full fit: mean `0.482%`, max `3.157%`
- cross-validation: mean `0.482%`, max `3.150%`

### `llama_8B`

- full fit: mean `0.866%`, max `3.473%`
- cross-validation: mean `0.866%`, max `3.473%`

### `llama_1B`

- full fit: mean `1.221%`, max `12.894%`
- cross-validation: mean `1.223%`, max `12.914%`

Interpretation:

- decode is largely solved for `3B` and `8B`
- `1B` decode is good on average but still has a long-tail max error problem

## Prefill Remains The Hard Part

Using corrected aggregation on the stable runs, prefill still did not reach the desired max-error target.

Stable corrected-aggregation prefill results:

### `llama_1B`

- full fit: mean `2.512%`, max `7.811%`

### `llama_3B`

- full fit: mean `3.906%`, max `9.468%`

### `llama_8B`

- full fit: mean `5.290%`, max `14.231%`

## Dense Prefill Sweep

Because prefill was under-sampled in the 9-point stable grid, a denser prefill-focused sweep was collected.

Settings:

- output length fixed at `17`
- input lengths: `127, 255, 511, 767, 1023, 1279, 1535`
- batch sizes: `1, 2, 4`
- warmup rounds: `2`
- measurement rounds: `5`

Outputs:

- `/users/rh/tmp/fresh_distserve_formula_fit/prefill_dense`

Analysis summary:

- `/users/rh/tmp/fresh_distserve_formula_fit/prefill_dense_analysis.json`

Conclusion from dense prefill sweep:

- even with denser coverage, one global low-order formula did not reach the target
- cubic terms help, but not enough

Best dense-prefill cross-validation results found:

### `llama_1B`

- mean `3.596%`
- max `9.156%`

### `llama_3B`

- mean `4.370%`
- max `12.599%`

### `llama_8B`

- mean `6.369%`
- max `14.597%`

## Lowest-Cost Deployable Model Chosen

The lowest-cost practical model family found so far is:

- decode: corrected aggregation formula
- prefill: batch-specific cubic fit from the dense prefill sweep

This bundle was written to:

- `/users/rh/tmp/fresh_distserve_formula_fit/chosen_low_cost_model.json`

Why this was chosen:

- smallest implementation cost among the remaining options
- simple to deploy
- uses measured data directly
- better than continuing to force one global prefill polynomial

Important limitation:

- this low-cost model still does not fully satisfy the original prefill target
- decode is mostly within target for `3B` and `8B`
- prefill remains above the desired max error

## Main Technical Conclusions

### 1. The remaining-token factor is not necessary for a good decode model

Once the decode target is constructed correctly, the best decode fits rely on:

- batch size
- current context length
- block/page-aligned context terms

### 2. Correct target construction matters more than adding arbitrary terms

The biggest improvements came from:

- better reruns
- more measurements
- corrected aggregation

not from blindly adding features

### 3. DistServe runtime instability affects data collection

Repeatedly observed on `llama_3B`:

- CUDA illegal-memory-access at the final config
- the profiler checkpointing change prevented loss of completed data

### 4. Prefill likely needs either a richer model class or lower-level profiling

The current single-forward benchmark plus low-order closed-form formulas are not enough to force prefill under:

- mean `2%`
- max `5%`

for all three models

## Recommended Next Step

If continuing, the next step with the highest expected value is not another small feature tweak.

Best next options:

1. Use the chosen low-cost model immediately for simulator work:
   - batch-specific cubic prefill
   - corrected decode formula
2. If strict prefill accuracy is mandatory, collect lower-level prefill traces or kernel-stage timings.
3. If deployment simplicity matters more than mathematical elegance, use interpolation or a table-based prefill model over `(batch_size, input_len)`.

At the current stage, option 1 is the cheapest workable path.

## Mixed-Batch Decode Follow-Up

After the concern about `bs_ctx`, a new decode-specific experiment was run on heterogeneous batches to test a more physical model.

Tmp scripts and outputs:

- profiler: `/users/rh/tmp/fresh_distserve_formula_fit/profile_distserve_mixed_decode.py`
- analysis: `/users/rh/tmp/fresh_distserve_formula_fit/analyze_mixed_decode.py`
- raw data: `/users/rh/tmp/fresh_distserve_formula_fit/mixed_decode/`
- analysis summary: `/users/rh/tmp/fresh_distserve_formula_fit/mixed_decode_analysis/mixed_decode_analysis.json`
- short write-up: `/users/rh/tmp/fresh_distserve_formula_fit/mixed_decode_analysis.md`
- recommended bundle: `/users/rh/tmp/fresh_distserve_formula_fit/chosen_mixed_decode_model.json`

### What Was Tested

Mixed prompt-length decode batches were profiled for:

- `llama_1B`
- `llama_3B`
- `llama_8B`

The comparison included:

- paper-style decode formula: `const + sum_context_len`
- old homogeneous-fit extension: `const + sum_context_len + max_context_blocks + sum_context_blocks`
- new heterogeneous formulas with `max_context_len`, `sum_context_len`, and block terms

Evaluation used:

- median decode time for each `(batch_template, decode_step)`
- relative least-squares fitting
- leave-one-batch-template-out validation

### Main Result

The pure paper-style decode formula does not work on these heterogeneous-batch measurements.

Leave-one-batch-template-out error for `const + sum_context_len`:

- `llama_1B`: mean `13.158%`, max `44.972%`
- `llama_3B`: mean `17.441%`, max `64.291%`
- `llama_8B`: mean `13.286%`, max `42.544%`

The best shared physically sensible decode formula is:

- `const + max_context_len + sum_context_len + max_context_blocks + sum_context_blocks`

Leave-one-batch-template-out error for that shared formula:

- `llama_1B`: mean `0.971%`, max `7.031%`
- `llama_3B`: mean `0.673%`, max `4.475%`
- `llama_8B`: mean `0.738%`, max `3.230%`

If per-model formulas are allowed, the best mean-error models are:

- `llama_1B`: `const + max_context_len + sum_context_len + sum_context_blocks`
- `llama_3B`: `const + batch_size + max_context_len + sum_context_len + max_context_blocks + sum_context_blocks`
- `llama_8B`: `const + batch_size + max_context_len + sum_context_len + max_context_blocks + sum_context_blocks`

Per-model best leave-one-batch-template-out error:

- `llama_1B`: mean `0.968%`, max `7.053%`
- `llama_3B`: mean `0.544%`, max `3.936%`
- `llama_8B`: mean `0.619%`, max `2.996%`

### Interpretation

This follow-up confirms that the criticism of `bs_ctx` was valid.

For mixed batches, a decode formula needs both:

- a `max_context_len` term to represent longest-sequence wall-clock gating
- a `sum_context_len` term to represent total KV-read / attention work

The block terms:

- `max_context_blocks`
- `sum_context_blocks`

help capture page/block-level KV-cache overhead.

### Updated Practical Recommendation

For the lowest-cost shared decode formula that still has a good physical interpretation, use:

- `const + max_context_len + sum_context_len + max_context_blocks + sum_context_blocks`

This replaces the earlier confidence in the homogeneous `bs_ctx` form as the preferred decode model for mixed batches.

Current status relative to the original target:

- `llama_3B` decode now meets the target comfortably
- `llama_8B` decode now meets the target comfortably
- `llama_1B` decode is still very good on average, but the max error tail remains around `7%`

So the main remaining obstacle is still:

- prefill accuracy across all three models
- and the `llama_1B` decode tail if the strict max-error target must hold everywhere

## Extra 1B Noise Check

Because the remaining decode gap was concentrated on `llama_1B`, the cheapest next check was a denser rerun of only the mixed-batch `1B` decode sweep.

Rerun location:

- raw: `/users/rh/tmp/fresh_distserve_formula_fit/mixed_decode_1b_more_rounds/llama_1B`
- analysis: `/users/rh/tmp/fresh_distserve_formula_fit/mixed_decode_1b_more_rounds_analysis/mixed_decode_analysis.json`

Settings:

- `warmup_rounds = 3`
- `measure_rounds = 15`

Expectation:

- if the earlier `~7%` max error was just under-sampling noise, more rounds should reduce the tail

Observed result:

- the rerun became much noisier instead of cleaner

Using the same analysis script, `llama_1B` leave-one-batch-template-out error for the best earlier mixed decode formulas degraded from about:

- mean `~0.97%`
- max `~7.0%`

to roughly:

- mean `36%` to `38%`
- max `67%` to `83%`

depending on the exact formula

Variance evidence:

- original 5-round sweep: mean per-point CV `2.337%`, p95 CV `4.758%`, max CV `45.997%`
- 15-round rerun: mean per-point CV `19.137%`, p95 CV `52.364%`, max CV `94.487%`

Interpretation:

- the long rerun was strongly affected by shared-environment noise
- in this environment, adding more rounds does not guarantee better targets
- the current strict max-error goal cannot be trusted unless the profiling environment is much more stable

Practical conclusion:

- the earlier mixed decode fit is still the better estimate
- but the unresolved `llama_1B` tail should now be treated as partly a data-collection problem, not only a formula problem

If strict `max <= 5%` is mandatory, the next required step is likely:

- dedicated GPU isolation for profiling
- or lower-level stage/kernel timing rather than end-to-end shared-GPU wall-clock timing
