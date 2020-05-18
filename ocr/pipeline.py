import time

import cv2
import torch

from detector.net import CRAFTDetector
from recognizer.net import CRNNRecognizer, MORANRecognizer

start = time.time()
uses = 'MORAN'
detector = CRAFTDetector()
if uses == 'CRNN':
    recognizer = CRNNRecognizer()
else:
    recognizer = MORANRecognizer()
init = time.time()
print(f'took {init-start:.3f}s for init')
detector.load()
recognizer.load()
recload = time.time()
print(f'took {recload-init:.3f}s for init')

img = 'img/test2.jpg'
res = []
processed = cv2.imread(img)

# detection
roi, _, _, _ = detector.process(processed)
detproc = time.time()
print(f'took {detproc-recload:.3f}s with CRAFT for detection. img:{img}\n')

# recognition
for _, img in enumerate(roi):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if uses == 'CRNN':
        text, _ = recognizer.process(gray)
    else:
        text, _, _ = recognizer.process(gray)
    res.append(text)

recproc = time.time()
print(f'took {recproc-detproc:.3f}s using {uses} for recognition')

for i in res:
    print(f'prediction: {i}')
