# RF-DETR ONNX models (drop files here)

Two exported RF-DETR detectors, trained on 2D architectural drawings.

Place, with these exact names (or tell me the real names):

    models/rfdetr/walls_windows.onnx      # Model A: walls + windows
    models/rfdetr/walls_windows.labels.json
    models/rfdetr/rooms_elements.onnx      # Model B: typed rooms + doors/elements
    models/rfdetr/rooms_elements.labels.json

labels.json = class-index -> name map, e.g.:
    {"0": "wall", "1": "window"}
    {"0": "bedroom", "1": "kitchen", "2": "toilet", "3": "living", "4": "door", ...}

I introspect input size / output tensors from the .onnx at load; I only need
the label map from you (ONNX doesn't carry class names).
