### ocr

_handle end-to-end scene text detection-recogntion_
- `/detector` holds models for text detection task while `/recognizer` contains models for text recognition
- `convert.py` converts model into .onnx
- `pipeline.py` connects `detector` `recognizer` for OCR
