from fastapi import FastAPI, UploadFile, File
from typing import List
from zipfile import ZipFile
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import tempfile, os

from roi_tracker import ROITracker
from sam2_utils import load_sam2_local

app = FastAPI()

sam2_predictor = load_sam2_local(predictor_type="video", model_size="large")
roi_tracker = ROITracker(sam2_predictor)

@app.post("/track")
async def track(
    video: UploadFile = File(...),
    points: List[str] = None,   # ["x,y", "x,y", ...]
    frame_idx: int = 0
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        video_path = tmp.name

    parsed_points = []
    for p in points or []:
        x, y = p.split(",")
        parsed_points.append((int(x), int(y)))

    prompt = {
        "frame_idx": frame_idx,
        "points": parsed_points,
        "labels": [1] * len(parsed_points),
    }

    masks, _ = roi_tracker.track(
        video_path=video_path,
        prompts=[prompt],
        save_to=None,
        verbose=False,
    )

    # Serialize masks into PNGs and pack into ZIP
    masks_dir = tempfile.mkdtemp()
    mask_paths = []

    for i, mask in enumerate(masks):
        mask_array = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_array)
        mask_path = os.path.join(masks_dir, f"mask_{i}.png")
        mask_img.save(mask_path)
        mask_paths.append(mask_path)

    zip_path = os.path.join(tempfile.gettempdir(), f"masks_{os.getpid()}.zip")
    with ZipFile(zip_path, "w") as zipf:
        for p in mask_paths:
            zipf.write(p, os.path.basename(p))
    os.remove(video_path)
    return FileResponse(zip_path, filename="masks.zip", media_type="application/zip")