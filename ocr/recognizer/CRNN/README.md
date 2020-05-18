### CRNN 
[paper](https://arxiv.org/pdf/1507.05717.pdf) | [original implementation](https://github.com/bgshih/crnn)

__architecture__: TPS-ResNet-biLSTM as encoder and a forward attention layer as decoder.

* __TODO__: added [ ICDAR2019 ](https://rrc.cvc.uab.es/?com=introduction) for val_set in conjunction with MJSynth data
* training: run ```python tool/generator.py``` to create dataset, `train.py` for training the models
