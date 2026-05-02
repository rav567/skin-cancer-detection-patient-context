import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

DATA_DIR = Path("datasets")
METADATA_PATH = DATA_DIR / "train-metadata.csv"
IMAGE_DIR = DATA_DIR / "train-image" / "image"

REQUIRED_COLUMNS = [
    "isic_id",
    "patient_id",
    "target",
    "age_approx",
    "sex",
    "anatom_site_general",
    "clin_size_long_diam_mm",
]

VALID_SEX_VALUES = {"male", "female"}


class LesionRecord(BaseModel):
    isic_id: str
    patient_id: str
    target: int = Field(ge=0, le=1)
    age_approx: Optional[float] = Field(default=None, ge=0, le=120)
    sex: Optional[str] = None
    anatom_site_general: Optional[str] = None
    clin_size_long_diam_mm: Optional[float] = Field(default=None, ge=0)

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in VALID_SEX_VALUES:
            raise ValueError(f"Unexpected sex value: {value}")
        return value


def check_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def check_no_nulls_in_critical_fields(df: pd.DataFrame) -> None:
    critical = ["isic_id", "patient_id", "target"]
    for column in critical:
        null_count = df[column].isna().sum()
        if null_count > 0:
            raise ValueError(f"Found {null_count} null values in critical field '{column}'")


def check_no_duplicate_ids(df: pd.DataFrame) -> None:
    duplicate_count = df["isic_id"].duplicated().sum()
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate isic_id values")


def check_target_values(df: pd.DataFrame) -> None:
    invalid_targets = df.loc[~df["target"].isin([0, 1]), "target"].unique()
    if len(invalid_targets) > 0:
        raise ValueError(f"Found invalid target values: {invalid_targets}")


def check_image_files_exist(df: pd.DataFrame, image_dir: Path = IMAGE_DIR, sample_size: int = 100) -> None:
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    missing = []
    for isic_id in sample["isic_id"]:
        image_path = image_dir / f"{isic_id}.jpg"
        if not image_path.exists():
            missing.append(isic_id)
    if missing:
        raise FileNotFoundError(
            f"Missing image files for {len(missing)} of {len(sample)} sampled IDs. "
            f"First few: {missing[:5]}"
        )


def validate_sample_records(df: pd.DataFrame, sample_size: int = 1000) -> None:
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    errors = []
    for record_dict in sample.to_dict(orient="records"):
        cleaned = {k: (None if pd.isna(v) else v) for k, v in record_dict.items() if k in LesionRecord.model_fields}
        try:
            LesionRecord(**cleaned)
        except ValidationError as exc:
            errors.append((record_dict.get("isic_id"), str(exc)))
    if errors:
        sample_errors = errors[:3]
        raise ValueError(f"Schema validation failed for {len(errors)} records. Examples: {sample_errors}")


def validate_metadata(metadata_path: Path = METADATA_PATH, image_dir: Path = IMAGE_DIR) -> pd.DataFrame:
    logger.info(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    logger.info("Checking required columns")
    check_required_columns(df)

    logger.info("Checking for nulls in critical fields")
    check_no_nulls_in_critical_fields(df)

    logger.info("Checking for duplicate IDs")
    check_no_duplicate_ids(df)

    logger.info("Checking target values")
    check_target_values(df)

    logger.info("Validating record schema on sample")
    validate_sample_records(df)

    logger.info("Verifying image files exist on sample")
    check_image_files_exist(df, image_dir=image_dir)

    logger.info("All validation checks passed")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = validate_metadata()
    print(f"\nValidation passed. Dataset has {len(df):,} records, {df['patient_id'].nunique():,} patients, {df['target'].sum()} malignant cases.")