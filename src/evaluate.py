import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, roc_curve


def bootstrap(y_true, y_scores, n_iterations, n_interp, seed):
    np.random.seed(seed)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    bootstrap_aurocs = []
    bootstrap_auprcs = []
    bootstrap_tprs = []
    bootstrap_precisions = []
    interp_points = np.linspace(0, 1, 500)

    # Perform bootstrap
    for _ in tqdm(range(n_iterations), desc="Performing bootstrap iteration"):
        # Generate random indices with replacement
        indices = np.random.randint(0, len(y_true), len(y_true))

        # Sample data using the generated indices
        y_true_sample = y_true[indices]
        y_scores_sample = y_scores[indices]

        # Calculate AUROC and AUPRC for the sample
        fpr, tpr, roc_thresholds = roc_curve(y_true_sample, y_scores_sample)
        auroc_score = auc(fpr, tpr)
        precision, recall, pr_thresholds = precision_recall_curve(
            y_true_sample, y_scores_sample
        )
        auprc_score = auc(recall, precision)

        # Interpolate ROC curve
        roc_curve_interp = np.interp(
            interp_points,
            fpr,
            tpr,
        )
        tpr = roc_curve_interp

        # Interpolate PR curve
        recall = np.flip(recall)
        precision = np.flip(precision)
        pr_curve_interp = np.interp(
            interp_points,
            recall,
            precision,
        )
        precision = pr_curve_interp

        # Append samples
        bootstrap_aurocs.append(auroc_score)
        bootstrap_auprcs.append(auprc_score)
        bootstrap_tprs.append(tpr)
        bootstrap_precisions.append(precision)

    return (
        np.array(bootstrap_aurocs),
        np.array(bootstrap_auprcs),
        interp_points,
        np.array(bootstrap_tprs),
        np.array(bootstrap_precisions),
    )


def mean_confidence_interval(values, confidence):
    mean = np.mean(values, axis=0)
    lower_ci = np.percentile(values, (100 - confidence) / 2, axis=0)
    upper_ci = np.percentile(values, 100 - (100 - confidence) / 2, axis=0)
    lower_ci = np.maximum(lower_ci, 0)
    upper_ci = np.minimum(upper_ci, 1)
    return mean, lower_ci, upper_ci


def evaluate(
    y_true,
    y_scores,
    n_bootstrap=1000,
    n_interp=1000,
    seed=42,
    confidence=95,
    verbose=False,
):
    # Perform boostrapping:
    (
        bootstrap_aurocs,
        bootstrap_auprcs,
        interp_points,
        bootstrap_tprs,
        bootstrap_precisions,
    ) = bootstrap(y_true, y_scores, n_bootstrap, n_interp, seed)

    # Calculate mean and 95% confidence intervals:
    mean_auroc, lower_ci_auroc, upper_ci_auroc = mean_confidence_interval(
        bootstrap_aurocs, confidence
    )
    mean_auprc, lower_ci_auprc, upper_ci_auprc = mean_confidence_interval(
        bootstrap_auprcs, confidence
    )
    mean_tpr, lower_ci_tpr, upper_ci_tpr = mean_confidence_interval(
        bootstrap_tprs, confidence
    )
    mean_precision, lower_ci_precision, upper_ci_precision = mean_confidence_interval(
        bootstrap_precisions, confidence
    )

    # Log metrics:
    if verbose:
        print(
            "AUROC: {:.2f} ({:.2f}, {:.2f})".format(
                mean_auroc, lower_ci_auroc, upper_ci_auroc
            )
        )
        print(
            "AUPRC: {:.2f} ({:.2f}, {:.2f})".format(
                mean_auprc, lower_ci_auprc, upper_ci_auprc
            )
        )

    results = {
        "metrics": {
            "mean_auroc": mean_auroc,
            "lower_auroc": lower_ci_auroc,
            "upper_auroc": upper_ci_auroc,
            "mean_auprc": mean_auprc,
            "lower_auprc": lower_ci_auprc,
            "upper_auprc": upper_ci_auprc,
        },
        "curves": {
            "fpr": interp_points.tolist(),
            "mean_tpr": mean_tpr.tolist(),
            "tpr_lower": lower_ci_tpr.tolist(),
            "tpr_upper": upper_ci_tpr.tolist(),
            "recall": interp_points.tolist(),
            "mean_precision": mean_precision.tolist(),
            "precision_lower": lower_ci_precision.tolist(),
            "precision_upper": upper_ci_precision.tolist(),
        },
    }
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AUROC/AUPRC for predicted risk scores file."
    )
    parser.add_argument("scores_file", help="Risk scores file.")
    return parser.parse_args()


def main():
    # Parse arguments:
    print("Parsing arguments ...")
    try:
        args = parse_args()
        scores_file = args.scores_file
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        raise

    # Load risk scores:
    with open(scores_file, "r") as f:
        risk_scores = json.load(f)
        y_true = risk_scores["y_true"]
        y_scores = risk_scores["y_scores"]

    # Evaluate predicted risk scores:
    print("Evaluating predicted risk scores ...")
    try:
        evaluate(y_true, y_scores, verbose=True)
    except Exception as e:
        print(f"Error parsing evaluating predicted risk scores: {e}")
        raise


if __name__ == "__main__":
    main()