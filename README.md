# lightly-ocr

[![CircleCI](https://circleci.com/gh/aar0npham/lightly-ocr/tree/master.svg?style=svg)](https://circleci.com/gh/aar0npham/lightly-ocr/tree/master)

lightly's backend - receipt to text :chart_with_downwards_trend:

OCR tasks can be found in `/ocr`, ingress controller can be found in `/ingress`

__NOTES__: _CRAFT_ and _MORAN_ are ported from to original repository with some compatibility fixes to work with newer version , refers to [credits.](#credits)

## table of content.
* [credits.](#credits)
* [structure.](#structure)
* [todo.](#todo)
* [how to use this repo.](#how-to-use-this-repo)
* [develop.](#develop)
* [tldr.](#tldr)

## credits.
* [MORAN-v2](https://github.com/Canjie-Luo/MORAN_v2)
* [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)
* [crnn.pytorch](https://github.com/meijieru/crnn.pytorch)

## todo.

* [ ] complete `__init__.py`
* [ ] add docstring, fixes `too-many-locals`
* [ ] backend controller for [ingress](ingress/)
* [ ] custom ops for `torch.nn.functional.grid_sample`
* [x] ~~added Dockerfile/CircleCI~~

<details>
<summary>
<a href="ocr/"><b>text-detection</b></a>
</summary><br>

- <b>CRAFT</b>
  * [ ] add `unit_test`
  * [ ] includes training loop (_under construction_)

- <b>YOLO</b>
  * [ ] updates training loops
</details>

<details>
<summary>
<a href="ocr/"><b>text-recognition</b></a>
</summary><br>

- <b>CRNN</b>
  * [ ] add `unit_test`
  * [ ] fixes `batch_first` for AttentionCell in [sequence.py](ocr/modules/sequence.py)
  * [ ] process ICDAR2019 for eval sets in conjunction with MJSynth val data ⇒ reduce biases
  * [x] ~~transfer trained weight to fit with the model~~
  * [x] ~~fix image padding issues with [eval.py](ocr/recognizer/CRNN/tools/eval.py)~~
  * [x] ~~creates a general dataset and generator function for both reconition model~~
  * [x] ~~database parsing for training loop~~
  * [x] ~~__FIXME__: gradient vanishing when training~~
  * [x] ~~generates logs for each training session~~
  * [x] ~~add options for continue training~~
  * [x] ~~modules incompatible shapes~~
  * [x] ~~create lmdb as dataset~~
  * [x] ~~added [generator.py](ocr/recognizer/CRNN/tools/generator.py) to generate lmdb~~
  * [x] ~~merges valuation_fn into [train.py](ocr/recognizer/CRNN/train.py#L136)~~

</details>

## structure.
overview in `src` as follows:
```bash
./
├── models              # location for model save
├── modules             # contains core file for setting up models
├── data                # contains training/testing/validation dataset
├── train               # code to train specific model
├── test                # contains unit testing file
├── tools               # contains tools to generate dataset/image processing etc.
├── config.yml          # config.yml 
├── convert.py          # convert model to .onnx file format
├── model.py            # contains model constructed from `modules`
├── net.py              # end-to-end OCR 
└── pipeline.py
```

## how to use this repo.
- This repo by no means to disregard/take credits from the work of original authors. I'm just having fun taking on the challenges and reimplement for `lightly`
- if you want to use MORAN please refer to [tree@66171c8058](https://github.com/aar0npham/lightly-ocr/tree/66171c80586537ae915938b2e92eb83c474cda79)
- Run `bash scripts/download_model.sh` to get the pretrained model
- to test the model do:
```python
python ocr/pipeline.py --img [IM_PATH]
```
- to train your own refers to [tldr.](#tldr). Only support CRNN atm, I will add CRAFT/YOLO/MORAN in the near future

## develop.
- make sure the repository is up to date with ```git pull origin master```
- create a new branch __BEFORE__ working with ```git checkout -b feats-branch_name``` where `feats` is the feature you are working on and the `branch-name` is the directory containing that features. 
  
  e.g: `git checkout -b YOLO-ocr` means you are working on `YOLO` inside `ocr`
- make your changes as you wish
- ```git status``` to show your changes and then following with ```git add .``` to _stage_ your changes
- commit these changes with ```git commit -m "describe what you do here"```

<details>
<summary>if you have more than one commit you should <i>rebase/squash</i> small commits into one</summary><br>

- ```git status``` to show the amount of your changes comparing to _HEAD_: 
  
  ```Your branch is ahead of 'origin/master' by n commit.``` where `n` is the number of your commit 
- ```git rebase -i HEAD~n``` to changes commit, __REMEMBER__ `-i`
- Once you enter the interactive shell `pick` your first commit and `squash` all the following commits after that
- after saving and exits edit your commit message once the new windows open describe what you did
- more information [here](https://git-scm.com/docs/git-rebase)

</details><br>

- push these changes with ```git push origin feats-branch_name```. do ```git branch``` to check which branch you are on
- then make a pull-request on github!
- have fun hacking!

## tldr. 

### [CRAFT.](ocr/net.py#L55)
[paper](https://arxiv.org/pdf/1904.01941.pdf) | [original implementation](https://github.com/clovaai/CRAFT-pytorch)

__architecture__: VGG16-Unet as backbone, with [UpConv](ocr/modules/vgg_bn.py#L23) return region/affinity score
* adopt from [ Faster R-CNN ](https://arxiv.org/pdf/1506.01497.pdf)

### [CRNN.](ocr/net.py#L211) 
[paper](https://arxiv.org/pdf/1507.05717.pdf) | [original implementation](https://github.com/bgshih/crnn)

__architecture__: TPS-ResNet-biLSTM as encoder and a forward attention layer as decoder.

* training: run ```python tools/generator.py``` to create dataset, `train/crnn.py` for training the models
* model is under `model.py`

