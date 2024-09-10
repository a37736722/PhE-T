import sys
sys.path.append("UKB-Tools")

import argparse
from tqdm import tqdm
from importlib.machinery import SourceFileLoader

from ukb_tools.data import UKB
from ukb_tools.logger import logger
from ukb_tools.preprocess.utils import rename_features
from ukb_tools.preprocess.labeling import match_phenotype
from ukb_tools.preprocess.filtering import filter_partially_populated_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data", help="Path to CSV file containing raw UKB data.")
    parser.add_argument("config", help="Path to the preprocessing config file.")
    parser.add_argument(
        "out_file",
        help="File to write the resulting dataframe.",
        default="preprocessed_data.csv",
        nargs="?",
        const=1,
    )
    return parser.parse_args()


def preprocess_pipeline(df, preprocess_cfg):
    # Unpack config vars:
    features = preprocess_cfg.features
    phenotypes_ids = preprocess_cfg.phenotype_ids

    # Filter participant with valid phenotype entries:
    eids_valid_pheno = filter_partially_populated_rows(df, phenotypes_ids)
    df = df.loc[list(eids_valid_pheno)]

    # Label the participant based on whether they have been diagnosed or not:
    matching_rules = preprocess_cfg.phenotype_matching_rules
    for phenotype, rules in tqdm(
        matching_rules.items(), desc="Labeling participant's phenotypes"
    ):
        df[f"{phenotype}"] = df.apply(match_phenotype, phenotype_rules=rules, axis=1)

    # Rename feature columns:
    features_dict = {feat.name: feat.field_id for feat in features}
    df, feature_names = rename_features(df, features_dict)

    # Keep features and phenotypes columns:
    phenotype_names = list(matching_rules.keys())
    df = df[feature_names + phenotype_names]

    # Compute average diastolic blood pressure:
    df["Diastolic blood pressure"] = (
        (df["Diastolic blood pressure_0"] + df["Diastolic blood pressure_1"])
    ) / 2
    df = df.drop(columns=["Diastolic blood pressure_0", "Diastolic blood pressure_1"])
    
    # Filter participant with valid features:
    for feat in tqdm(features, desc="Filtering features"):
        df = df[df[feat.name].apply(feat.is_valid)]
    
    # Decode features:
    for feat in tqdm(features, desc="Decoding features"):
        df[feat.name] = df[feat.name].apply(
            lambda x: feat.decode_map[x] if feat.decode_map else x
        )

    return df


def main():
    # Parse arguments:
    args = parse_args()
    raw_data = args.raw_data
    out_file = args.out_file
    preprocess_cfg = SourceFileLoader("config", args.config).load_module()

    # Load data:
    logger.info("Loading UKB data ...")
    ukb = UKB(raw_data)
    ukb.load_data(instance="0")

    # Preprocess data:
    ukb.preprocess(preprocess_pipeline, [preprocess_cfg])

    # Save to CSV file:
    logger.info(f"Saving data to {out_file} ...")
    ukb.data.to_csv(out_file)
    logger.info("Data saved successfully.")


if __name__ == "__main__":
    main()