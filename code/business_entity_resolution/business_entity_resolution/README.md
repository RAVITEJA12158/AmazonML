# Business Entity Resolution — v3 Pipeline

Status: **code complete, wired for your real student_resource dataset, LightGBM only** (`pair_model.candidates: ["lightgbm"]` in config.yaml — no CatBoost/XGBoost). `data/raw/` no longer contains any placeholder/synthetic data — it's empty until the adapter script below populates the real split.

## What changed in this pass

The previous split script (`13_split_student_train.py`) only ever split `source1` + `ground_truth`. It never touched `source2`/`source3` at all, and used a column-name convention (`business_name`/`business_address`) that the rest of the pipeline didn't know how to read. That's fixed now with a proper two-step flow:

1. **`scripts/13a_adapt_student_data.py`** — the only script that needs to know your real column names (`entity_id`, `business_name`, `business_address`, `country`, and ground truth's `matched_entity_ids`). Reads `student_data.dataset_train_dir` **only** (never the sibling `test` folder), and writes canonical-schema files to `artifacts/student_data/adapted/`.
2. **`scripts/13_split_student_train.py`** — splits **only** `source1` + `labels` into two partitions, grouped by `source1_entity_id` so no entity's candidates straddle the split. `source2`/`source3` are **not duplicated** — both partitions share the one candidate pool via `student_data.shared_candidates_dir`, which is both correct (holding out S1 queries shouldn't remove their true matches from the candidate universe) and avoids copying a potentially huge file twice.

```
artifacts/splits/student_holdout/
  dev/            source1.tsv + labels.tsv   <- config.yaml paths.raw_dir points here; all tuning happens against this
  heldout_test/   source1.tsv + labels.tsv   <- scored exactly once, by scripts/14, never used for tuning
```

`dev_fraction: 0.5` in config.yaml splits it into two literal halves, per your instruction — change to `0.8` for a more conventional 80/20 dev/heldout split if you'd rather tune on more data.

The real, unlabeled hackathon **test** directory (`dataset_test_dir`) is opened by exactly one script — `scripts/15_score_real_test.py` — and only for final submission scoring, never for anything that could influence a modeling decision.

## Commands — run these in order, on your machine

```bash
pip install -r requirements.txt

# 1) One-time: adapt your real files (edit config.yaml's student_data.dataset_train_dir
#    first if it isn't already C:\Users\tsnvr\Documents\ML\amazon\AmazonML\code\business_entity_resolution\student_resource\dataset\train)
python scripts/13a_adapt_student_data.py

# 2) One-time: split into dev / heldout_test (never touches dataset_test_dir)
python scripts/13_split_student_train.py

# 3) Train + tune against dev/ (GroupKFold OOF at every stage — repeat this as much as you want)
python scripts/run_all.py

# 4) Run exactly once, when you're done tuning: honest score on data the model never saw
python scripts/14_evaluate_heldout.py

# 5) Final: score the real unlabeled test set for submission (also just one output file, no metric)
python scripts/15_score_real_test.py
```

If your real column/file names differ even slightly from what `13a_adapt_student_data.py` expects, it fails loudly and tells you which column it couldn't find — edit `FILE_ALIASES`/`COLUMN_ALIASES` at the top of that one file; nothing else needs to change.

### Smoke-testing the code without your real data

```bash
python scripts/demo_generate_synthetic_data.py   # writes FAKE data to data/demo_raw/, never data/raw/
# then temporarily point config.yaml paths.raw_dir at "data/demo_raw" and run scripts/run_all.py
```
This is for verifying the code runs at all — the metrics it produces are meaningless (they're on fake data) and should never be reported as real accuracy.

## Architecture — the restructured funnel

The core change from earlier drafts of this pipeline: **semantic retrieval never sees the full S2/S3 pool.**

```
S1 entity
   │
   ▼
Geographic pre-blocking (country/state/city, cheapest filter, first)
   │
   ▼
Lexical channels within the geo pool, unioned
   (token / char n-gram / phonetic / rare-token-IDF / concatenated name+address)
   │
   ▼
Cheap ranker — logistic regression, NOT hand-written weights
   (trained on cheap features: token overlap, jaro-winkler, address overlap,
    postal/city exact, phonetic match — falls back to a fixed heuristic only
    when there's too little labeled data to fit a stable model)
   │
   ▼
Semantic retrieval — ONLY on the ranker-reduced pool
   (pluggable backend: TF-IDF char n-gram cosine runs locally by default;
    Qwen3-Embedding-8B is stubbed — see "Known limitation" below)
   │
   ▼
Reciprocal-retrieval flags + candidate pruning (final per-entity size cap —
   candidate_pairs.tsv size is itself a scored objective, not just recall)
   │
   ▼
candidate_pairs.tsv
   │
   ▼
Pairwise feature engineering (name/address/cross-field/missingness/
   contradiction/uniqueness/source-aware/cross-source/reciprocal/margin)
   │
   ▼
Pair model (LightGBM) → 5-fold GroupKFold OOF predictions
   │
   ▼
OOF hard-negative mining (5 categories) → weight swept via re-run OOF, not guessed
   │
   ▼
Final pair model, retrained on hard-negative-weighted data
   │
   ▼
Calibration (Platt / isotonic) — kept only if it beats raw scores on OOF
   │
   ▼
Entity-level decision: hand-rule baseline vs. learned LightGBM model,
   compared on OOF macro F0.5, winner kept — never forces top-1 per entity
   │
   ▼
matching_results.tsv  (every S1 entity gets a row — a NONE marker if no
   match was accepted, so singleton predictions aren't silently absent)
```

## What's implemented and verified

| Script | What it does | Verified output on synthetic data |
|---|---|---|
| `00_generate_sample_data.py` | Synthetic S1/S2/S3 + labels with legal-suffix variants, address noise, near-duplicate hard negatives, singletons | 15 S1 / 61 S2+S3 / 21 labeled pairs |
| `03_build_candidates.py` | Fits the cheap ranker on labeled data, runs the full funnel for every entity | `candidate_pairs.tsv`: 186 rows, 12.4 candidates/entity avg |
| `04_candidate_diagnostics.py` | Recall ceiling, per-entity size stats, channel contribution | **recall_ceiling = 1.0**, full per-channel breakdown |
| `05_build_features.py` | Full pairwise feature table | 186 rows × 76 columns |
| `06_train_pair_oof.py` | 5-fold GroupKFold OOF training + diagnostic threshold sweep | OOF macro F0.5 = 0.9698 (pre-hard-negative) |
| `07_mine_hard_negatives.py` | Mines 5 hard-negative categories from OOF, **sweeps the sample weight via actual re-run OOF training** (not a random guess) | 35 hard negatives mined (mostly category E: same-location-different-business) |
| `08_train_final_pair_model.py` | Trains final pair model on all dev data with hard-negative weighting | Saved + feature importances logged |
| `09_calibrate_and_threshold.py` | Tests Platt/isotonic against raw OOF scores, keeps the winner, freezes the threshold | **Isotonic genuinely won** (0.9889 vs. 0.9698 raw) — demonstrates "kept only if it helps" is real, not decorative |
| `10_train_entity_model.py` | Hand-rule baseline vs. learned entity model, compared on OOF macro F0.5 | Learned model won on this run (1.0 vs. 0.9667) — real A/B comparison, not a coin flip |
| `11_inference.py` | Full frozen pipeline, no test-time tuning | 15/15 entities represented, `candidate_pairs.tsv` + `matching_results.tsv` written |
| `12_validate_submission.py` | Checks coverage, duplicates, orphaned matches, valid source labels | **Passes on good output, genuinely fails when a broken row is injected** (tested) |

## Known limitation: Qwen3-Embedding-8B

This sandbox has no internet access to model-hosting sites and no GPU, so the real target embedding model can't run here. `src/blocking/semantic.py` implements a pluggable `EmbeddingBackend` interface:

- `TfidfBackend` (default, `semantic_retrieval.embedding_backend: "tfidf"` in `config.yaml`) — character n-gram TF-IDF + cosine similarity, runs anywhere, no external calls. This is what actually ran in every test above.
- `QwenBackend` — stubbed, raises a clear `NotImplementedError` with instructions rather than silently falling back. To use it: install `sentence-transformers`, download the model weights in your own environment, implement `QwenBackend.embed()`, and flip the config flag. Nothing else in the funnel (union → prune → features) needs to change.

## Other things worth knowing before you run this on real data

- **`cheap_ranker` fallback fired on the synthetic data** (only 10 positive examples, below the `min_positive_examples: 20` threshold in config) — it used the fixed-weight heuristic instead of a fitted logistic regression. On real data with more labeled pairs, confirm in the logs that it actually fits the model rather than silently falling back.
- **Hard-negative category D (near-duplicate name clusters)** didn't fire on the synthetic set — category E (same postal/city, different business) dominated instead. This is a property of the synthetic generator, not a pipeline bug; check the category breakdown on real data since the methodology docs flag near-duplicate names as the highest-value category to catch.
- **`geo_pool` and the per-entity funnel are implemented per-entity in Python**, not with prebuilt global inverted indices for the geo stage specifically (the lexical/phonetic/rare-token channels do use global indices via `build_token_idf` and the channel functions). For true billion-record scale, replace `src/blocking/geographic.py`'s per-entity DataFrame filter with a proper partitioned/bucketed lookup (e.g., a hash index or a Spark partition keyed on country/state/city) — flagged here rather than silently shipped as if it already scales.
- **`src/features/cross_source.py`'s S2↔S3 matching is O(n²) within each entity's candidate set** — fine at ~40 candidates/entity (the configured cap), would need indexing if `max_candidates_per_entity` is raised substantially.

## Bugs found and fixed during this build (for transparency)

1. `add_candidate_relative_features` used `groupby(...).apply()` with a function that mutated and returned the group frame — this silently dropped the `source1_entity_id` column in this pandas version. Fixed by switching to `groupby(...).transform()`, which can't lose the grouping column. Caught because script 05's output was checked for the column's presence, not just its shape.
2. `postal_code`/`street_number` round-tripping through CSV as floats (`94105.0`) broke exact-match features. Fixed with a `_clean_code()` helper in `src/normalization/addresses.py`.
3. Hard-negative mining referenced a `raw_score` column that OOF training actually produces as `oof_score` — column-name mismatch caught on first run of script 07, fixed by aligning the names.
4. The hard-negative sample weight was originally a single random draw from the configured range — replaced with an actual 3-point OOF re-training sweep (script 07) so the selected weight is empirically justified, not guessed.
5. `matching_results.tsv` originally only contained accepted-match rows, which meant singleton entities (correctly predicted as "no match") were silently absent from the file — indistinguishable from a forgotten entity. Fixed with explicit `NONE` marker rows (`write_matching_results` in `src/pipeline/inference.py`) and the validator updated to treat them as valid.
6. Building the entity-level threshold sweep initially used `rename(columns={"entity_score": "oof_score"})` on a frame that already had an `oof_score` column, producing duplicate column labels and a reindex crash. Fixed by constructing the scored frame explicitly instead of renaming in place.

## Repository structure

See `src/` for implementation (organized by `data/`, `normalization/`, `blocking/`, `features/`, `models/`, `mining/`, `evaluation/`, `pipeline/`, `utils/`) and `scripts/` for the numbered, runnable pipeline stages plus `run_all.py`.
