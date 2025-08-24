from dataclasses import dataclass
import pandas as pd


@dataclass
class DataFormat:
    ts_event: str
    bid_fill: float
    ask_fill: float
    price: float
    best_bid: float
    best_ask: float
    mid_price: float


def validate_data_format(df: pd.DataFrame):
    """
    Validates that a DataFrame conforms to the DataFormat schema.

    Args:
        df (pd.DataFrame): The DataFrame to validate.

    Raises:
        TypeError: If a column has an incorrect data type.
        ValueError: If a column is missing.
    """
    for field, field_type in DataFormat.__annotations__.items():
        if field not in df.columns:
            raise ValueError(f"Missing column in DataFrame: '{field}'")

        actual_type = df[field].dtype

        if field_type is float and not pd.api.types.is_numeric_dtype(actual_type):
            raise TypeError(f"Column '{field}' should be numeric, but is {actual_type}")
        elif field_type is str and not (
            pd.api.types.is_string_dtype(actual_type)
            or pd.api.types.is_object_dtype(actual_type)
        ):
            raise TypeError(
                f"Column '{field}' should be a string, but is {actual_type}"
            )


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file and validates it against the DataFormat schema.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded and validated DataFrame.
    """
    df = pd.read_csv(file_path)
    validate_data_format(df)
    return df
