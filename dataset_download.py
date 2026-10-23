import os
from pathlib import Path
from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")

if not api_key:
    raise RuntimeError(
        "ROBOFLOW_API_KEY is not set. Set it in the terminal, then rerun."
    )

download_dir = Path("tree-classification-v1").resolve()

rf = Roboflow(api_key=api_key)

project = (
    rf.workspace("aadhya-agarwal")
    .project("tree-classification-hrkkh")
)

dataset = project.version(1).download(
    model_format="folder",
    location=str(download_dir),
    overwrite=True,
)

print("\nDownload complete!")
print(f"Dataset saved to: {dataset.location}")
