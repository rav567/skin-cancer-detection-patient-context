import logging
import time
import zipfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

COMPETITION_NAME = "isic-2024-challenge"
DATA_DIR = Path("datasets")
EXPECTED_FILES = ["train-metadata.csv", "train-image"]
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 30


def data_already_exists(data_dir: Path = DATA_DIR) -> bool:
    if not data_dir.exists():
        return False
    return all((data_dir / name).exists() for name in EXPECTED_FILES)


def authenticate_kaggle():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def download_competition_files(api, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    api.competition_download_files(COMPETITION_NAME, path=str(target_dir), quiet=False)


def extract_zip(target_dir: Path) -> None:
    zip_path = target_dir / f"{COMPETITION_NAME}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected zip not found at {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)
    zip_path.unlink()


def verify_download(target_dir: Path) -> None:
    missing = [name for name in EXPECTED_FILES if not (target_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Download verification failed. Missing: {missing}")


def download_dataset(force: bool = False) -> None:
    if not force and data_already_exists():
        logger.info("Dataset already present. Skipping download.")
        return

    logger.info("Beginning Kaggle dataset download.")
    api = authenticate_kaggle()

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Download attempt {attempt} of {MAX_RETRIES}.")
            download_competition_files(api, DATA_DIR)
            extract_zip(DATA_DIR)
            verify_download(DATA_DIR)
            logger.info("Download and extraction successful.")
            return
        except Exception as error:
            last_error = error
            logger.warning(f"Attempt {attempt} failed: {error}")
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds.")
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(f"Download failed after {MAX_RETRIES} attempts.") from last_error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    download_dataset()