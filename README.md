# lightly-ocr

lightly's backend - from receipt to text on browser o.o

Anything that deals with ocr will can be found in `/src`, related to backend controller will can be found in `/ingress`

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
./
├── models
├── detection
│   ├── CRAFT
│   │   ├── backbone.py
│   │   ├── craft_utils.py
│   │   ├── imgproc.py
│   │   └── model.py
│   └── YOLO
├── recognition
│   ├── CRNN
│   │   ├── datagen.py
│   │   ├── models.py
│   │   └── utils.py
│   └── MORAN
│       ├── asrn_resnet.py
│       ├── fractional_pickup.py
│       ├── model.py
│       ├── morn.py
│       └── test.py
├── detector.py
├── pipeline.py
├── convert.py
├── recognizer.py
```

## instruction.
