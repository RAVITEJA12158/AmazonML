"""Runs the model-training pipeline (03-12) against whatever
`paths.raw_dir` currently points at in config.yaml.

This does NOT adapt or split the real student_resource data -- run those
once, first:
    python scripts/13a_adapt_student_data.py
    python scripts/13_split_student_train.py
(config.yaml's paths.raw_dir already points at the dev/ split they produce)

For a quick no-real-data smoke test instead, use:
    python scripts/demo_generate_synthetic_data.py
    python scripts/run_all.py

Run: python scripts/run_all.py
"""
import subprocess
import sys
import time

STEPS = [
    "scripts/03_build_candidates.py",
    "scripts/04_candidate_diagnostics.py",
    "scripts/05_build_features.py",
    "scripts/06_train_pair_oof.py",
    "scripts/07_mine_hard_negatives.py",
    "scripts/08_train_final_pair_model.py",
    "scripts/09_calibrate_and_threshold.py",
    "scripts/10_train_entity_model.py",
    "scripts/11_inference.py",
    "scripts/12_validate_submission.py",
]


def main():
    print("=" * 70)
    print("Business Entity Resolution — full pipeline run")
    print("=" * 70)

    for step in STEPS:
        print(f"\n--- {step} ---")
        t0 = time.time()
        result = subprocess.run([sys.executable, step])
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"\n!!! {step} FAILED (exit {result.returncode}) after {elapsed:.1f}s — stopping.")
            sys.exit(result.returncode)
        print(f"({elapsed:.1f}s)")

    print("\n" + "=" * 70)
    print("Pipeline completed successfully. See output/matching_results.tsv "
          "and output/candidate_pairs.tsv.")
    print("=" * 70)


if __name__ == "__main__":
    main()
