# lightly-ocr

lightly's backend - from receipt to text on browser o.o

Anything that deals with ocr will can be found in `/src`, related to backend controller will can be found in `/ingress`, model is saved in `/models`
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
├── detection                 # responsible for text classifier/detection
│   ├── CRAFT                 # contains codebase for CRAFT                    
│   │   ├── data_loader.py    # load data          
│   │   ├── loss.py           # loss functions for CRAFT (wip)
│   │   ├── mep.py            # minimal enclosing parallelogram for classifier 
│   │   ├── model.py          # CRAFT model in pytorch
│   │   ├── modules/          # contains backbone for CRAFT
│   │   └── utils/            # contains utils for img processing
│   └── YOLO                  # contains codebase for YOLOv3
├── recognition               # reponsible for text recognition
│   ├── CRNN                  # contains codebase for CRNN
│   │   ├── logs/             # training logs
│   │   ├── config.yml        # configuration for training
│   │   ├── mjsynth.sh        # downloading ECCV 2014
│   │   ├── model.py          # CRNN model in tensorflow
│   │   └── utils.py          # utility func
│   └── MORAN                 # MORAN in pytorch
│       ├── model.py          # MORAN model in pytorch
│       ├── test.py           # test code
│       ├── modules/          # backbone for MORAN
│       └── utils/            # contains utils files
├── pipeline.ipynb            # chain into OCR 
├── detector.py               # running first half task : detection
├── recognizer.py             # running second half task : recognition 
└── convert.py                # convert model weights -> .onnx -> tfjs [direct conversion for .h5]
```

## instruction.
