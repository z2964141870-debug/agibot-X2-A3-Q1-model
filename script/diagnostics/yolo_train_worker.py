"""One-shot Ultralytics worker for the ordinary-YOLO reboot probe."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def mark(path: Path, phase: str, **extra: object) -> None:
    rec = {"at": time.time(), "phase": phase, **extra}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> int:
    out = Path(os.environ["YOLO_PROBE_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    phases = out / "worker_phases.jsonl"
    model_path = os.environ["YOLO_MODEL"]
    data_path = os.environ["YOLO_DATA"]
    epochs = int(os.environ.get("YOLO_EPOCHS", "20"))
    batch = int(os.environ.get("YOLO_BATCH", "32"))
    imgsz = int(os.environ.get("YOLO_IMGSZ", "640"))
    project = os.environ["YOLO_PROJECT"]
    name = os.environ["YOLO_NAME"]
    try:
        mark(phases, "IMPORT_BEGIN")
        from ultralytics import YOLO  # type: ignore
        mark(phases, "IMPORT_END")
        mark(phases, "MODEL_LOAD_BEGIN", model=model_path)
        model = YOLO(model_path)
        mark(phases, "MODEL_LOAD_END")
        mark(phases, "TRAIN_BEGIN", epochs=epochs, batch=batch, imgsz=imgsz)
        result = model.train(
            data=data_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=0,
            workers=4,
            cache=False,
            amp=False,
            pretrained=False,
            project=project,
            name=name,
            exist_ok=False,
            verbose=True,
            plots=False,
            save_period=1,
        )
        mark(phases, "TRAIN_END", result_type=type(result).__name__)
        return 0
    except BaseException as exc:
        mark(phases, "ERROR", error_type=type(exc).__name__, error=str(exc)[-2000:])
        raise


if __name__ == "__main__":
    sys.exit(main())
