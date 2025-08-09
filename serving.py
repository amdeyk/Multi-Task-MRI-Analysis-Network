"""Model serving infrastructure for MRI-KAN."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from config import Config
from optimized_network import SOTAMRINetwork
from monitoring import setup_logging

app = FastAPI(title="MRI-KAN Serving")
logger = setup_logging()

MODEL: Optional[torch.nn.Module] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionResponse(BaseModel):
    segmentation: list[int]


@app.on_event("startup")
def load_model() -> None:  # pragma: no cover - executed at runtime
    """Load model weights at application start."""
    global MODEL
    cfg = Config()
    MODEL = SOTAMRINetwork(cfg)
    ckpt_path = Path("checkpoints/best.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=DEVICE)
        MODEL.load_state_dict(state.get("model", state))
        logger.info("Loaded model checkpoint from %s", ckpt_path)
    MODEL.to(DEVICE)
    MODEL.eval()


@app.post("/infer", response_model=PredictionResponse)
async def infer(file: UploadFile) -> PredictionResponse:
    """Run inference on a single MRI volume provided as a NumPy array."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        contents = await file.read()
        volume = np.load(file.file if hasattr(file, "file") else file, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read input: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid input") from exc
    with torch.no_grad():
        tensor = torch.from_numpy(volume).unsqueeze(0).to(DEVICE)
        outputs = MODEL(tensor)
        pred = outputs["segmentation"].argmax(dim=1).cpu().numpy().tolist()[0]
    return PredictionResponse(segmentation=pred)
