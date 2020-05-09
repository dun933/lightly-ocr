import io
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, TensorBoard
import numpy as np
import cv2
from PIL import Image
from .gaussian import GaussianGenerator
from .data import load_data
from .box import reorder_points
from .img import load_sample, img_unnormalize, load_image, img_normalize
from .fake import crop_image, watershed, find_box, un_warping, cal_confidence, divide_region, enlarge_char_boxes

class SampleGenerator(Callback):
    def __init__(self, base_model, train_sample_lists, train_sample_probs, fakes, img_size, batch_size):
        super().__init__()
        assert len(train_sample_lists) == len(train_sample_probs)
        assert len(train_sample_lists) == len(fakes)
        self.base_model = base_model
        self.train_sample_lists = train_sample_lists
        self.fakes = fakes
        self.train_sample_probs = np.array(train_sample_probs) / np.sum(train_sample_probs)
        self.sample_count_list = [len(sample_list) for sample_list in train_sample_lists]
        self.sample_idx_list = [0] * len(train_sample_lists)
        self.sample_mark_list = list(range(len(train_sample_lists)))
        self.img_size = img_size
        self.batch_size = batch_size

    def get_batch(self, size, is_true=True):
        images = list()
        word_boxes_list = list()
        word_lengths_list = list()
        region_scores = list()
        affinity_scores = list()
        confidence_score_list = list()
        fg_masks = list()
        bg_masks = list()
        gaussian_generator = GaussianGenerator()

        word_count_list = list()
        for i in range(size):
            if is_true:
                sample_mark = np.random.choice(self.sample_mark_list, p=self.train_sample_probs)
            else:
                while 1:
                    sample_mark = np.random.choice(self.sample_mark_list, p=self.train_sample_probs)
                    if self.fakes[sample_mark]:
                        break
            img_path, word_boxes, words, char_boxes_list, confidence_list = \
                self.train_sample_lists[sample_mark][self.sample_idx_list[sample_mark]]
            self.sample_idx_list[sample_mark] += 1
            if self.sample_idx_list[sample_mark] >= self.sample_count_list[sample_mark]:
                self.sample_idx_list[sample_mark] = 0
                np.random.shuffle(self.train_sample_lists[sample_mark])

            img, word_boxes, char_boxes_list, region_box_list, affinity_box_list, img_shape = \
                load_sample(img_path, self.img_size, word_boxes, char_boxes_list)

            images.append(img)

            word_count = min(len(word_boxes), len(words))
            word_boxes = np.array(word_boxes[:word_count], dtype=np.int32) // 2
            word_boxes_list.append(word_boxes)
            word_count_list.append(word_count)

            word_lengths = [len(words[j]) if len(char_boxes_list[j]) == 0 else 0 for j in range(word_count)]
            word_lengths_list.append(word_lengths)

            height, width = img.shape[:2]
            heat_map_size = (height // 2, width // 2)

            mask_shape = (img_shape[1] // 2, img_shape[0] // 2)
            confidence_score = np.ones(heat_map_size, dtype=np.float32)
            for word_box, confidence_value in zip(word_boxes, confidence_list):
                if confidence_value == 1:
                    continue
                tmp_confidence_score = np.zeros(heat_map_size, dtype=np.uint8)
                cv2.fillPoly(tmp_confidence_score, [np.array(word_box)], 1)
                tmp_confidence_score = np.float32(tmp_confidence_score) * confidence_value
                confidence_score = \
                    np.where(tmp_confidence_score > confidence_score, tmp_confidence_score, confidence_score)
            confidence_score_list.append(confidence_score)

            fg_mask = np.zeros(heat_map_size, dtype=np.uint8)
            cv2.fillPoly(fg_mask, [np.array(word_box) for word_box in word_boxes], 1)
            fg_masks.append(fg_mask)
            bg_mask = np.zeros(heat_map_size, dtype=np.float32)
            bg_mask[:mask_shape[0], :mask_shape[1]] = 1
            bg_mask = bg_mask - fg_mask
            bg_mask = np.clip(bg_mask, 0, 1)
            bg_masks.append(bg_mask)

            region_score = gaussian_generator.gen(heat_map_size, np.array(region_box_list) // 2)
            region_scores.append(region_score)

            affinity_score = gaussian_generator.gen(heat_map_size, np.array(affinity_box_list) // 2)
            affinity_scores.append(affinity_score)

        max_word_count = np.max(word_count_list)
        max_word_count = max(1, max_word_count)
        new_word_boxes_list = np.zeros((size, max_word_count, 4, 2), dtype=np.int32)
        new_word_lengths_list = np.zeros((size, max_word_count), dtype=np.int32)
        for i in range(size):
            if word_count_list[i] > 0:
                new_word_boxes_list[i, :word_count_list[i]] = np.array(word_boxes_list[i])
                new_word_lengths_list[i, :word_count_list[i]] = np.array(word_lengths_list[i])

        images = np.array(images)
        region_scores = np.array(region_scores, dtype=np.float32)
        affinity_scores = np.array(affinity_scores, dtype=np.float32)
        confidence_scores = np.array(confidence_score_list, dtype=np.float32)
        fg_masks = np.array(fg_masks, dtype=np.float32)
        bg_masks = np.array(bg_masks, dtype=np.float32)

        inputs = {
            'image': images,
            'word_box': new_word_boxes_list,
            'word_length': new_word_lengths_list,
            'region': region_scores,
            'affinity': affinity_scores,
            'confidence': confidence_scores,
            'fg_mask': fg_masks,
            'bg_mask': bg_masks,
        }

        outputs = {
            'craft': np.zeros([size])
        }

        return inputs, outputs

    def fake_char_boxes(self, src, word_box, word_length):
        img, src_points, crop_points = crop_image(src, word_box, dst_height=64.)
        h, w = img.shape[:2]
        if min(h, w) == 0:
            confidence = 0.5
            region_boxes = divide_region(word_box, word_length)
            region_boxes = [reorder_points(region_box) for region_box in region_boxes]
            return region_boxes, confidence
        img = img_normalize(img)
        # print(img.shape)
        region_score, _ = self.base_model.predict(np.array([img]))
        heat_map = region_score[0] * 255.
        heat_map = heat_map.astype(np.uint8)
        marker_map = watershed(heat_map)
        region_boxes = find_box(marker_map)
        confidence = cal_confidence(region_boxes, word_length)
        if confidence <= 0.5:
            confidence = 0.5
            region_boxes = divide_region(word_box, word_length)
            region_boxes = [reorder_points(region_box) for region_box in region_boxes]
        else:
            region_boxes = np.array(region_boxes) * 2
            region_boxes = enlarge_char_boxes(region_boxes, crop_points)
            region_boxes = [un_warping(region_box, src_points, crop_points) for region_box in region_boxes]
            # print(word_box, region_boxes)

        return region_boxes, confidence

    def init_sample(self, flag=False):
        for sample_mark in self.sample_mark_list:
            if self.fakes[sample_mark]:
                sample_list = self.train_sample_lists[sample_mark]
                new_sample_list = list()

                for sample in sample_list:
                    if len(sample) == 5:
                        img_path, word_boxes, words, _, _ = sample
                    else:
                        img_path, word_boxes, words, _ = sample
                    img = load_image(img_path)
                    char_boxes_list = list()

                    confidence_list = list()
                    for word_box, word in zip(word_boxes, words):
                        char_boxes, confidence = self.fake_char_boxes(img, word_box, len(word))
                        char_boxes_list.append(char_boxes)
                        confidence_list.append(confidence)
                    new_sample_list.append([img_path, word_boxes, words, char_boxes_list, confidence_list])

                self.train_sample_lists[sample_mark] = new_sample_list
            elif flag:
                sample_list = self.train_sample_lists[sample_mark]
                new_sample_list = list()

                for sample in sample_list:
                    if len(sample) == 5:
                        img_path, word_boxes, words, char_boxes_list, _ = sample
                    else:
                        img_path, word_boxes, words, char_boxes_list = sample
                    confidence_list = [1] * len(word_boxes)
                    new_sample_list.append([img_path, word_boxes, words, char_boxes_list, confidence_list])

                self.train_sample_lists[sample_mark] = new_sample_list

    def on_epoch_end(self, epoch, logs=None):
        self.init_sample()

    def on_train_begin(self, logs=None):
        self.init_sample(True)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.batch_size)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.batch_size, False)
            yield ret


class CRAFTensorBoard(TensorBoard):
    def __init__(self, log_dir, write_graph, test_model, callback_model, data_generator):
        self.test_model = test_model
        self.callback_model = callback_model
        self.data_generator = data_generator
        self.writers = tf.summary.create_file_writer('../logs')
        super(CRAFTensorBoard, self).__init__(log_dir=log_dir, write_graph=write_graph)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'learning_rate': K.eval(self.model.optimizer.lr)})
        with self.writers.as_default():
            data = next(self.data_generator.next_val())
            images = data[0]['image']
            word_boxes = data[0]['word_box']
            word_lengths = data[0]['word_length']
            target_region = data[0]['region']
            target_affinity = data[0]['affinity']
            confidence_scores = data[0]['confidence']
            region, affinity, region_gt, affinity_gt = self.callback_model.predict([images, word_boxes, word_lengths,
                                                                                    target_region, target_affinity,
                                                                                    confidence_scores])
            summaries = []
            for i in range(3):
                summaries.append(tf.summary.image(name=f'input_image/{i}', data=img_unnormalize(images[i])))
                summaries.append(tf.summary.image(name=f'region_pred/{i}', data=(region[i] * 255).astype('uint8')))
                summaries.append(tf.summary.image(name=f'affinity_pred/{i}', data=(affinity[i] * 255).astype('uint8')))
                summaries.append(tf.summary.image(name=f'region_gt/{i}', data=(region_gt[i] * 255).astype('uint8')))
                summaries.append(tf.summary.image(name=f'affinity_gt/{i}', data=(affinity_gt[i] * 255).astype('uint8')))
        super(CRAFTensorBoard, self).on_epoch_end(epoch + 1, logs)
        self.test_model.save_weights(f'models/weight_{epoch}.h5')
