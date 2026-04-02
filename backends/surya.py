"""
Surya OCR backend.

Uses Surya recognition model with full-image bbox to skip the detection step,
which is unnecessary when the input is already cropped to the subtitle region.
"""

from PIL import Image


def load():
    """Load and return the Surya recognition predictor."""
    from surya.recognition import RecognitionPredictor
    from surya.foundation import FoundationPredictor
    from surya.settings import settings

    print("Loading Surya recognition model...")
    foundation = FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)
    return RecognitionPredictor(foundation)


def run(images: list[Image.Image], state) -> list[str]:
    """Run Surya OCR on a batch of cropped images, return text per image."""
    from surya.common.surya.schema import TaskNames

    # Pass a full-image bbox per image to bypass the detection step
    full_bboxes = [[[0, 0, img.width, img.height]] for img in images]

    results = state(
        images,
        task_names=[TaskNames.ocr_without_boxes] * len(images),
        bboxes=full_bboxes,
    )

    texts = []
    for ocr_result in results:
        lines = [line.text for line in ocr_result.text_lines if line.text.strip()]
        texts.append("\n".join(lines))
    return texts
