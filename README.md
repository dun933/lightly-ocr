# ohcr
Three words: hackathon, OCR, EAST =)

## Content
* [Credits](#credits)
* [Description](#description)
* [Todoist](#todoist)
* [Notes](#notes)

## Description
- written in Keras
- Text Recognition
  - 4 layers of STR model that follows
      | Transform layer | Feature Extraction | Sequence Labeling    | Encoder                 |
      | --------------- | ------------------ | -------------------- | ----------------------- |
      | STN             | ResNet50/Darknet53 | biLSTM/AttentionLSTM | CTC/Joint CTC-Attention |

## Todoist
* ~~__URGENT__: Improve runtime for linking different file~~
* [x] nmt dataset (_finished on feb 14_)
* [x] construction of SHA-RNN (_finished on feb 14_)
* [x] construction of YOLO (_finished on march 24_)
* [x] Chain up them models for bigger chain (_finished on march 25_)
* [ ] Added CTC Loss

## Notes
- refers to [readme](helper/README)
## Credits
* [clovaai's deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
