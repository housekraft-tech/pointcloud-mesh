import os
import cv2
import pytest
from scripts.experiments.rfdetr_infer import (
    load, run_model, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS)

ROOM_CLASSES = {"Bedroom", "Bathroom", "Kitchen", "Utility", "Walkin",
                "Dining Room", "Balcony", "Living Room", "Foyer"}
PLAN = "floorplan_original.png"


@pytest.mark.skipif(not os.path.exists("models/rfdetr_elements.onnx"),
                    reason="RF-DETR elements model weights not present")
def test_elements_model_detects_rooms_on_koushik_plan():
    bgr = cv2.imread(PLAN)
    assert bgr is not None, f"{PLAN} not found"
    dets = run_model(load("elements"), bgr, ELEMENTS_CLASSES,
                     ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS)
    rooms = [d for d in dets if d["name"] in ROOM_CLASSES]
    assert len(rooms) >= 6, f"expected >=6 room boxes, got {len(rooms)}: {[d['name'] for d in rooms]}"
