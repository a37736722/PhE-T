import sys

sys.path.append(".")
sys.path.append("..")

import os
import json
import argparse
from src.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to evaluate predicted risk scores and generate results."
    )
    parser.add_argument("scores_dir", help="Risk scores directory.")
    parser.add_argument("result_dir", help="Directory to store results.")
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number bootstrapping iteration",
    )
    parser.add_argument(
        "--n_interp",
        type=int,
        default=100,
        help="Number interpolation point for ROC and PR curves",
    )
    parser.add_argument(
        "--confidence",
        type=int,
        default=95,
        help="Confidence of the confidence intervals",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed",
    )
    return parser.parse_args()


def main():
    # Parse arguments:
    print("Parsing arguments ...")
    try:
        args = parse_args()
        scores_dir = args.scores_dir
        result_dir = args.result_dir
        n_bootstrap = args.n_bootstrap
        n_interp = args.n_interp
        confidence = args.confidence
        seed = args.seed
    except Exception as e:
        print(f"Error parsing arguments: {e}")

    # List risk scores files
    for file in os.listdir(os.path.join(scores_dir)):
        # Load risk scores
        with open(os.path.join(scores_dir, file), "r") as f:
            risk_scores = json.load(f)
            y_true = risk_scores["y_true"]
            y_scores = risk_scores["y_scores"]

        # Evaluate predicted risk scores
        print("Evaluating predicted risk scores ...")
        try:
            results = evaluate(
                y_true,
                y_scores,
                n_bootstrap=n_bootstrap,
                n_interp=n_interp,
                seed=seed,
                confidence=confidence,
                verbose=True,
            )
        except Exception as e:
            print(f"Error parsing evaluating predicted risk scores: {e}")
            raise
        
        # Save results
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, file.split('rs_')[1]), "w") as f:
            f.write(json.dumps(results['metrics']))


if __name__ == "__main__":
    main()