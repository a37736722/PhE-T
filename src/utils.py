import pandas as pd
from typing import Callable
from dataclasses import dataclass


@dataclass
class Feature:
    name: str
    field_id: str
    is_valid: Callable
    unit: str = None
    decode_map: dict = None


def check_icd_code(value, code_list):
    return any(value.startswith(prefix) for prefix in code_list)


def is_valid_date(value):
    return not (
        pd.isna(value)
        or value
        in [
            "1900-01-01",
            "1901-01-01",
            "1902-02-02",
            "1903-03-03",
            "1909-09-09",
            "2037-07-07",
        ]
    )