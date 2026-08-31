
from paddleocr import PaddleOCR


def _iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    intersection = inter_w * inter_h

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


class OCRParser:

    def __init__(self):

        self.ocr_ar = PaddleOCR(
            lang="ar",
            enable_mkldnn=False
        )

        self.ocr_en = PaddleOCR(
            lang="en",
            enable_mkldnn=False
        )

    def _run_single_pass(
        self,
        image_path,
        ocr_engine
    ):

        results = ocr_engine.predict(image_path)

        texts = []

        for result in results:

            rec_texts = result["rec_texts"]
            rec_scores = result["rec_scores"]
            rec_boxes = result["rec_boxes"]

            for text, score, box in zip(
                rec_texts,
                rec_scores,
                rec_boxes
            ):

                x1, y1, x2, y2 = box.tolist()

                texts.append({
                    "text": text,
                    "confidence": float(score),
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ]
                })

        return texts

    def _merge_passes(
        self,
        ar_texts,
        en_texts,
        iou_threshold=0.5
    ):

        merged = list(ar_texts)

        used_ar_indices = set()

        for en_item in en_texts:

            best_match_index = None
            best_iou = 0.0

            for i, ar_item in enumerate(ar_texts):

                if i in used_ar_indices:
                    continue

                overlap = _iou(
                    en_item["bbox"],
                    ar_item["bbox"]
                )

                if overlap > best_iou:

                    best_iou = overlap
                    best_match_index = i

            if (
                best_match_index is not None
                and best_iou >= iou_threshold
            ):

                ar_item = ar_texts[best_match_index]

                if (
                    en_item["confidence"]
                    > ar_item["confidence"]
                ):
                    merged[best_match_index] = en_item

                used_ar_indices.add(
                    best_match_index
                )

            else:

                merged.append(en_item)

        return merged

    def parse(self, image_path):

        ar_texts = self._run_single_pass(
            image_path,
            self.ocr_ar
        )

        en_texts = self._run_single_pass(
            image_path,
            self.ocr_en
        )

        merged_texts = self._merge_passes(
            ar_texts,
            en_texts
        )

        return {
            "texts": merged_texts
        }
