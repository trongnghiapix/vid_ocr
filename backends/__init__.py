"""
Backend interface for OCR engines.

Each backend module must expose two functions:

    load() -> Any
        Load and return the model/predictor state.

    run(images: list[PIL.Image.Image], state: Any) -> list[str]
        Run OCR on a batch of images, return one text string per image.
        Multi-line text within an image is joined with newline.
"""
