import pandas as pd
from src.utils import Feature, check_icd_code, is_valid_date


features = [
    Feature(name="Age", field_id="21003", unit="years", is_valid=lambda x: x >= 0),
    Feature(
        name="Sex",
        field_id="31",
        unit=None,
        is_valid=lambda x: x in [0, 1],
        decode_map={0: "Male", 1: "Female"},
    ),
    Feature(name="BMI", field_id="21001", unit="kg/m²", is_valid=lambda x: x >= 0),
    Feature(
        name="HDL cholesterol",
        field_id="30760",
        unit="mmol/L",
        is_valid=lambda x: x >= 0,
    ),
    Feature(
        name="LDL cholesterol",
        field_id="30780",
        unit="mmol/L",
        is_valid=lambda x: x >= 0,
    ),
    Feature(
        name="Total cholesterol",
        field_id="30690",
        unit="mmol/L",
        is_valid=lambda x: x >= 0,
    ),
    Feature(
        name="Triglycerides", field_id="30870", unit="mmol/L", is_valid=lambda x: x >= 0
    ),
    Feature(
        name="Diastolic blood pressure",
        field_id="4079",
        unit="mmHg",
        is_valid=lambda x: x >= 0,
    ),
    Feature(
        name="Ever smoked",
        field_id="20160",
        unit=None,
        is_valid=lambda x: x in [0, 1],
        decode_map={0: "No", 1: "Yes"},
    ),
    Feature(
        name="Snoring",
        field_id="1210",
        unit=None,
        is_valid=lambda x: x in [1, 2],
        decode_map={1: "Yes", 2: "No"},
    ),
    Feature(
        name="Insomnia",
        field_id="1200",
        unit=None,
        is_valid=lambda x: x in [1, 2, 3],
        decode_map={1: "Never/rarely", 2: "Sometimes", 3: "Usually"},
    ),
    Feature(
        name="Daytime napping",
        field_id="1190",
        unit=None,
        is_valid=lambda x: x in [1, 2, 3],
        decode_map={1: "Never/rarely", 2: "Sometimes", 3: "Usually"},
    ),
    Feature(
        name="Sleep duration",
        field_id="1160",
        unit="hours/day",
        is_valid=lambda x: x >= 0,
    ),
    Feature(
        name="Chronotype",
        field_id="1180",
        unit=None,
        is_valid=lambda x: x in [1, 2, 3, 4],
        decode_map={
            1: "Definitely a 'morning' person",
            2: "More a 'morning' than 'evening' person",
            3: "More an 'evening' than a 'morning' person",
            4: "Definitely an 'evening' person",
        },
    ),
]


phenotype_ids = ["41271", "41270", "20002"]  # ICD-9, ICD-10 and self-report
phenotype_matching_rules = {
    "Asthma": [
        ("41271", lambda x: check_icd_code(x, ["493"])),  # ICD-9
        ("41270", lambda x: check_icd_code(x, ["J45", "J46"])),  # ICD-10
        ("20002", lambda x: x in [1111]),  # Self-reported
    ],
    "Cataract": [
        ("41270", lambda x: check_icd_code(x, ["H25", "H26"])),  # ICD-10
        ("4700", lambda x: x >= 0),  # Age cataract diagnosed
        ("131164", lambda x: is_valid_date(x)),  # Date H25 first reported
        ("131166", lambda x: is_valid_date(x)),  # Date H26 first reported
        ("131165", lambda x: not pd.isna(x)),  # Source of report of H25 first reported
        ("131167", lambda x: not pd.isna(x)),  # Source of report of H26 first reported
    ],
    "Diabetes": [
        ("41271", lambda x: check_icd_code(x, ["250"])),  # ICD-9
        (
            "41270",
            lambda x: check_icd_code(x, ["E10", "E11", "E12", "E13", "E14"]),
        ),  # ICD-10
        ("20002", lambda x: x in [1220, 1222, 1223]),  # Self-reported
        ("2443", lambda x: x == 1),  # Diabetes diagnosed by doctor
    ],
    "GERD": [
        ("41270", lambda x: check_icd_code(x, ["K20", "K21"])),  # ICD-10
        ("131584", lambda x: is_valid_date(x)),  # Date K21 first reported
    ],
    "Hay-fever & Eczema": [
        (
            "41270",
            lambda x: check_icd_code(x, [f"L2{i}" for i in range(10)] + ["L30"]),
        ),  # ICD-10
        ("3761", lambda x: x >= 0),  # Age hay fever, rhinitis or eczema diagnosed
    ],
    "Major depression": [
        ("41270", lambda x: check_icd_code(x, ["F32", "F33"])),  # ICD-10
        ("20126", lambda x: x in [3, 4, 5]),  # Bipolar and major depression status
    ],
    "Myocardial infarction": [
        ("41271", lambda x: check_icd_code(x, ["410", "412", "4109", "4129"])),  # ICD-9
        ("41270", lambda x: check_icd_code(x, ["I21", "I252", "Z034"])),  # ICD-10
        ("20002", lambda x: x in [1075]),  # Self-reported
        ("6150", lambda x: x == 1),  # Vascular/heart problems diagnosed by doctor
        ("131298", lambda x: is_valid_date(x)),  # Date I21 first reported
        ("131299", lambda x: not pd.isna(x)),  # Source of report of I21 first reported
    ],
    "Osteoarthritis": [
        (
            "41270",
            lambda x: check_icd_code(x, ["M15", "M16", "M17", "M18", "M19"]),
        ),  # ICD-10
        ("131868", lambda x: is_valid_date(x)),  # Date M15 first reported
        ("131870", lambda x: is_valid_date(x)),  # Date M16 first reported
        ("131872", lambda x: is_valid_date(x)),  # Date M17 first reported
        ("131876", lambda x: is_valid_date(x)),  # Date M19 first reported
        ("131869", lambda x: not pd.isna(x)),  # Source of report of M15 first reported
        ("131871", lambda x: not pd.isna(x)),  # Source of report of M16 first reported
        ("131873", lambda x: not pd.isna(x)),  # Source of report of M17 first reported
        ("131877", lambda x: not pd.isna(x)),  # Source of report of M19 first reported
    ],
    "Pneumonia": [
        (
            "41271",
            lambda x: check_icd_code(
                x, ["401", "480", "481", "482", "483", "484", "486"]
            ),
        ),  # ICD-9
        (
            "41270",
            lambda x: check_icd_code(
                x, ["J12", "J13", "J14", "J15", "J16", "J17", "J18"]
            ),
        ),  # ICD-10
    ],
    "Stroke": [
        ("41271", lambda x: check_icd_code(x, ["434.91"])),  # ICD-9
        ("41270", lambda x: check_icd_code(x, ["I63", "I64"])),  # ICD-10
        ("6150", lambda x: x == 3),  # Vascular/heart problems diagnosed by doctor
        ("131368", lambda x: is_valid_date(x)),  # Date I64 first reported
        ("131369", lambda x: not pd.isna(x)),  # Source of report of I64 first reported
    ],
}