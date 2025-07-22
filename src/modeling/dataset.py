from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from roboflow import Roboflow

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

rf = Roboflow(api_key="EwT5UrQw7iDNxx1mC9Ob")


def download_from_roboflow(workspace: str, project: str, version: int, format: str = "yolov8"):
    project = rf.workspace("sam64-t4u3d").project("manga-translator-detection-r1kli")
    version = project.version(version)
    dataset = version.download(format)


def download_manga_on_english():
    workspace = "sam64-t4u3d"
    project = "manga-translator-detection-r1kli"
    version = 1
    format = "yolov8"
    download_from_roboflow(workspace, project, version, format)


def download_manga_on_japanese():
    workspace = "mangaseer"
    project = "manga-text-detection-xyvbw"
    version = 3
    format = "yolov8"
    download_from_roboflow(workspace, project, version, format)





@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    download_manga_on_japanese()
