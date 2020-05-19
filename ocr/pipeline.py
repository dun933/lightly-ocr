import os
import time

import cv2
import torch

from ocrnet import CRAFTDetector, CRNNRecognizer, MORANRecognizer

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
print(f'took {recload-init:.3f}s for `load()`')

img = 'test/test2.jpg'
res = []
processed = cv2.imread(img)

with torch.no_grad():
    # detection
    roi, _, _, _ = detector.process(processed)
    detproc = time.time()
    print(f'took {detproc-recload:.3f}s with CRAFT for detection. img: {img}')

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
with open(os.path.join(os.path.dirname(os.path.relpath(__file__)), 'results.txt'), 'w') as f:
    for i in res:
        f.write(f'prediction: {i}\n')
    f.close()
end = time.time()
print(f'took total of {end-start:.3f}s for the whole process')
torch.cuda.empty_cache()
