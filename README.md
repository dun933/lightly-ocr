# lightly-ocr

lightly's backend - from receipt to text on browser o.o

Anything that deals with ocr will can be found in `/src`, related to backend controller will can be found in `/ingress`, model is saved in `/models`

_NOTES_: codebase for _CRAFT_ and _MORAN_ are ported from to original repository with some compatibility fixes to work with newer version , refers to [credits.](#credit)

## table of content.
* [credits.](#credits)
* [structure.](#structure)
* [instruction.](#instruction)

## credits.
* [MORAN-v2](https://github.com/Canjie-Luo/MORAN_v2)
* [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)
* [clovaai's deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

## structure.
overview in `src` as follows:
```bash
├── detection
│   ├── CRAFT
│   │   ├── data_loader.py
│   │   ├── loss.py
│   │   ├── mep.py
│   │   ├── model.py
│   │   ├── modules
│   │   │   ├── backbone.py
│   │   │   └── refinenet.py
│   │   ├── notebook
│   │   │   └── eval.ipynb
│   │   └── utils
│   │       ├── coordinates.py
│   │       ├── CRAFT.py
│   │       ├── file.py
│   │       ├── gaussian.py
│   │       ├── imgproc.py
│   │       └── watershed.py
│   └── YOLO
│       ├── config.yml
│       ├── model.py
│       ├── modules
│       │   ├── crf.py
│       │   ├── darknet.py
│       │   └── layers.py
│       ├── preprocess
│       │   ├── coco_text.py
│       │   ├── cocotext.v2.json
│       │   ├── dataset.py
│       │   ├── exclude.yml
│       │   ├── generator.py
│       └── receipt_classifier.py
├── recognition
│   ├── CRNN
│   │   ├── config.yml
│   │   ├── model.py
│   │   ├── preprocess.sh
│   │   └── utils.py
│   └── MORAN
│       ├── model.py
│       ├── modules
│       │   ├── asrn_resnet.py
│       │   ├── fractional_pickup.py
│       │   └── morn.py
│       ├── test.py
│       └── utils
│           ├── dataset.py
│           └── tools.py
├── pipeline.ipynb
├── detector.py
├── convert.py
└── recognizer.py
```

## instruction.
