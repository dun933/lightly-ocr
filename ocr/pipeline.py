import cv2
import torch

from detector.net import CRAFTDetector
from recognizer.net import CRNNRecognizer, MORANRecognizer

uses = 'CRNN'
detector = CRAFTDetector()
if uses == 'CRNN':
    recognizer = CRNNRecognizer
else:
    recognizer = MORANRecognizer()
detector.load()
recognizer.load()

img = 'img/test.png'
res = []
processed = cv2.imread(img)

roi, _, _, _ = detector.process(processed)
for _, img in enumerate(roi):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if uses == 'CRNN':
        text, _ = recognizer.process(gray)
    else:
        text, _, _ = recognizer.process(gray)
    res.append(text)

print(res)
