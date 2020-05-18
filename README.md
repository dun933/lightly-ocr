# lightly-ocr

lightly's backend - from receipt to text on browser o.o

OCR tasks can be found in `/ocr`, ingress controller can be found in `/ingress`

_NOTES_: codebase for _CRAFT_ and _MORAN_ are ported from to original repository with some compatibility fixes to work with newer version , refers to [credits.](#credits)

## table of content.
* [credits.](#credits)
* [structure.](#structure)
* [todo.](#todo)
* [instruction.](#instruction)

## credits.
* [MORAN-v2](https://github.com/Canjie-Luo/MORAN_v2)
* [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)
* [pytorch bgshih's CRNN](https://github.com/meijieru/crnn.pytorch)

## todo.
<details>
<summary>
<a href="ocr/recognizer"><b>text-recognition</b></a>
</summary><br>
- <b>CRNN</b> 
  * [ ] transfer pretrained weight to fit with the model
  * [ ] fix image padding issues with [eval.py](ocr/recognizer/CRNN/tools/eval.py)
  * [ ] process ICDAR2019 for eval sets in conjunction with MJSynth val data ⇒ reduce biases
  * [ ] creates a general dataset and generator function for both reconition model
  * ~~[x] database parsing for training loop~~
  * [x] ~~__FIXME__: gradient vanishing when training~~
  * [x] ~~generates logs for each training session~~
  * [x] ~~add options for continue training~~
  * [x] ~~modules incompatible shapes~~
  * [x] ~~create lmdb as dataset~~
  * [x] ~~added [generator.py](ocr/recognizer/CRNN/tools/generator.py) to generate lmdb~~
  * [x] ~~merges valuation_fn into [train.py](ocr/recognizer/CRNN/train.py#L136)~~
- <b>MORAN</b>
  * [ ] updates Variable to Tensor since torch.autograd.Variable is deprecated
</details>


## structure.
overview in `src` as follows:
```bash
./
├── detector
│   ├── net.py
│   ├── CRAFT
│   │   ├── craft_utils.py
│   │   ├── imgproc.py
│   │   ├── model.py
│   │   └── vgg_bn.py
│   ├── models
│   └── YOLO
├── recognizer
│   ├── CRNN
│   │   ├── data
│   │   ├── modules
│   │   │   ├── backbone.py
│   │   │   ├── sequence.py
│   │   │   └── transform.py
│   │   ├── tools
│   │   │   ├── dataset.py
│   │   │   ├── generator.py
│   │   │   └── utils.py
│   │   ├── config.yml
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── test.py
│   │   └── train.py
│   ├── models
│   ├── MORAN
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── modules
│   │   │   ├── asrn_resnet.py
│   │   │   ├── fractional_pickup.py
│   │   │   └── morn.py
│   │   ├── test.py
│   │   └── utils.py
│   └── net.py
├── README.md
├── convert.py
├── pipeline.py
└── zoo.ipynb
```

## instruction.
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


