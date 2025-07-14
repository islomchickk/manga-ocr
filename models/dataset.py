from roboflow import Roboflow
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
