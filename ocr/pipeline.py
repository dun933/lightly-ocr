import torch
import cv2

from detector import CRAFTDetector
from recognizer import MORANRecognizer

detector = CRAFTDetector()
recognizer = MORANRecognizer()
detector.load()
recognizer.load()

img = '../test/test.png'
res = []
processed = cv2.imread(img)

roi,_,_,_ = detector.process(processed)
for _, img in enumerate(roi):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text, _, _ = recognizer.process(gray)
    res.append(text)

print(res)
