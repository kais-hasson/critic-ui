from pathlib import Path
from urllib.request import urlretrieve

from ultralytics import YOLO


MODEL_URL = (
    "https://huggingface.co/docling-project/ScreenParser/"
    "resolve/main/best.pt?download=true"
)

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "best.pt"
)


class ScreenParser:

    def __init__(
        self,
        model_path: str | None = None
    ):

        if model_path is None:
            model_path = str(DEFAULT_MODEL_PATH)

        model_path = Path(model_path)

        if not model_path.exists():

            print("ScreenParser model not found.")
            print("Downloading best.pt from Hugging Face...")

            model_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            urlretrieve(
                MODEL_URL,
                model_path
            )

            print(
                f"Model downloaded to: {model_path}"
            )

        self.model = YOLO(
            str(model_path)
        )

    def parse(
        self,
        image_path: str,
        min_confidence: float = 0.01
    ):

        results = self.model.predict(
            image_path,
            imgsz=1920,
            conf=min_confidence,
            iou=0.01,
        )

        detections = []

        for result in results:

            for box, cls_id, conf in zip(
                result.boxes.xyxy,
                result.boxes.cls,
                result.boxes.conf
            ):

                x1, y1, x2, y2 = box.tolist()

                width = x2 - x1
                height = y2 - y1
                area = width * height

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                detections.append({
                    "class": self.model.names[int(cls_id)],
                    "confidence": float(conf),
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ],
                    "width": float(width),
                    "height": float(height),
                    "area": float(area),
                    "center": [
                        float(center_x),
                        float(center_y)
                    ]
                })

        print("\nDetected elements BEFORE post-processing:")
        print("-" * 100)

        self.print_detections(detections)

        detections = self.remove_duplicates(
            detections
        )

        print("\nDetected elements AFTER post-processing:")
        print("-" * 100)

        self.print_detections(detections)

        print("-" * 100)

        return {
            "detections": detections
        }

    def print_detections(self, detections):

        for i, detection in enumerate(
            detections,
            start=1
        ):

            print(
                f"{i}. "
                f"Class: {detection['class']} | "
                f"Confidence: "
                f"{detection['confidence']:.3f} | "
                f"BBox: "
                f"{detection['bbox']} | "
                f"Size: "
                f"{detection['width']:.1f}x"
                f"{detection['height']:.1f}"
            )

    def calculate_iou(
        self,
        box1,
        box2
    ):

        x1 = max(
            box1[0],
            box2[0]
        )

        y1 = max(
            box1[1],
            box2[1]
        )

        x2 = min(
            box1[2],
            box2[2]
        )

        y2 = min(
            box1[3],
            box2[3]
        )

        intersection_width = max(
            0,
            x2 - x1
        )

        intersection_height = max(
            0,
            y2 - y1
        )

        intersection = (
            intersection_width *
            intersection_height
        )

        area1 = (
            (box1[2] - box1[0]) *
            (box1[3] - box1[1])
        )

        area2 = (
            (box2[2] - box2[0]) *
            (box2[3] - box2[1])
        )

        union = (
            area1 +
            area2 -
            intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    def remove_duplicates(
        self,
        detections
    ):

        detections = sorted(
            detections,
            key=lambda d: d["confidence"],
            reverse=True
        )

        result = []

        for current in detections:

            is_duplicate = False

            for existing in result:

                iou = self.calculate_iou(
                    current["bbox"],
                    existing["bbox"]
                )

                if iou >= 0.001:
                    is_duplicate = True
                    break

            if not is_duplicate:
                result.append(current)

        return result