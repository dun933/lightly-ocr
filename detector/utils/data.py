import pickle
import os
import cv2
import numpy as np
import os
import re
import codecs
import scipy.io as sio

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dir_path, dir_names, file_names) in os.walk(in_path):
        for file in file_names:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dir_path, file))
    return img_files, mask_files, gt_files


def save_results(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
    """ save text detection result one by one
    Args:
        img_file (str): image file name
        img (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
    Return:
        None
    """
    img = np.array(img)

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    # result directory
    # res_file = dirname + "res_" + filename + '.txt'
    res_img_file = dirname + "res_" + filename + '.jpg'

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # f = open(res_file, 'w')
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        # strResult = ','.join([str(p) for p in poly]) + '\r\n'
        # f.write(strResult)

        poly = poly.reshape(-1, 2)
        cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        # ptColor = (0, 255, 255)
        # if verticals is not None:
        #     if verticals[i]:
        #         ptColor = (255, 0, 0)

        if texts is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            cv2.putText(img, "{}".format(texts[i]), (poly[0][0] + 1, poly[0][1] + 1), font, font_scale, (0, 0, 0),
                        thickness=1)
            cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

    # Save result image
    cv2.imwrite(res_img_file, img)

class ICDAR2013Convertor(object):
    def  __init__(self, img_root, gt_root):
        super(ICDAR2013Convertor, self).__init__()
        self.img_root = img_root
        self.gt_root = gt_root

    def to_CRAFT(self):
        img_name_list = os.listdir(self.img_root)
        sample_list = list()
        for img_name in img_name_list:
            gt_name = f'gt_{img_name[:-len(img_name.split(".")[-1])]}txt'
            gt_path = os.path.join(self.gt_root, gt_name)
            img_path = os.path.join(self.img_root, img_name)
            if os.path.exists(gt_path):
                word_boxes = list()
                char_boxes_list = list()
                words = list()
                with codecs.open(gt_path, 'rb', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        infos = re.split(',? ', line)
                        word = infos[4]
                        word = re.sub('^"','',word)
                        word = re.sub('"$', '', word)
                        if '\\' in word:
                            print(word)
                        words.append(word)
                        char_boxes_list.append([])
                        left, top, right, bottom = [round(float(p)) for p in infos[:4]]
                        word_box = np.array([[left,top],[right,top], [right,bottom], [left, bottom]])
                        word_boxes.append(word_box)
                sample_list.append([img_path, word_boxes, words, char_boxes_list])
        return sample_list

class SynthTextConvertor(object):
    def __init__(self, mat_path, image_root):
        super(SynthTextConvertor, self).__init__()
        self.math_path = mat_path
        self.image_root = image_root
        self.image_name_list, self.word_boxes_list, self.char_boxes_list, self.texts_list = self.__load_mat()

    def __load_mat(self):
        data = sio.loadmat(self.math_path)
        image_name_list = data['imnames'][0]
        word_boxes_list = data['wordBB'][0]
        char_boxes_list = data['charBB'][0]
        texts_list = data['txt'][0]

        return image_name_list, word_boxes_list, char_boxes_list, texts_list

    @staticmethod
    def split_text(texts):
        split_texts = list()
        for text in texts:
            text = re.sub(' ','',text)
            split_texts += text.split()
        return split_texts

    @staticmethod
    def swap_box_axes(boxes):
        if len(boxes.shape) == 2 and boxes.shape[0] ==2 and boxes.shape[1]==4:
            # (2,4) -> (1,4,2)
            boxes = np.array([np.swapaxes(boxes, axis1=0, axis2=1)])
        else:
            # (2,4,n) -> (n,4,2)
            boxes = np.swapaxes(boxes, axis1=0, axis2=2)
        return boxes

    def to_CRAFT(self):
        sample_list = list()
        for img_name, word_boxes, char_boxes, texts in zip(self.image_name_list, self.word_boxes_list, self.char_boxes_list, self.texts_list):
            word_boxes = self.swap_box_axes(word_boxes)
            char_boxes = self.swap_box_axes(char_boxes)
            texts = self.split_text(texts)
            tmp_char_boxes_list = list()
            char_idx = 0
            for text in texts:
                char_count = len(text)
                tmp_char_boxes_list.append(char_boxes[char_idx:char_idx+char_count])
                char_idx+=char_count
            image_path = os.path.join(self.image_root, img_name[0])
            sample_list.append([image_path, word_boxes, texts, tmp_char_boxes_list])
        return sample_list
