import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *

from .modules.darknet import darknet53
from .modules.layers import conv2d_unit

working_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(working_path, 'config.yml'), 'r') as s:
    params = yaml.safe_load(s)

yolo_anchors = np.array([[18, 19], [34, 19], [47, 21], [34, 35], [62, 22], [52, 34], [78, 25], [97, 27], [68, 39], [52, 54], [87, 43], [116, 34], [140, 30], [75, 63], [105, 57], [169, 40], [137, 56], [91, 90], [204, 46], [126, 84], [169, 76], [248, 54], [147, 124], [305, 60], [212, 94], [275, 111], [367, 88], [209, 164], [472, 103], [336, 198]], np.float32) / params["IMAGE_SIZE"]
yolo_anchor_masks = np.array([range(20, 30), range(10, 20), range(0, 10)])


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def precision(y_true, y_pred, valid_detections, threshold):
    def precision_per_batch(label, pred, valid_detection):
        if valid_detection == 0:
            return 0

        TP = 0
        FP = 0

        for i in range(valid_detection):
            p = pred[i]
            iou_score = max(iou(l, p) for l in label)
            if iou_score > threshold:
                # pred bbox has correct detection
                TP += 1
            else:
                # pred bbox has incorrect detection
                FP += 1

        return float(TP / (TP + FP))

    return np.mean([precision_per_batch(label, pred, valid_detection) for label, pred, valid_detection in
                    zip(y_true, y_pred, valid_detections)])


def recall(y_true, y_pred, valid_detections, threshold):
    def recall_per_batch(label, pred, valid_detection):
        if valid_detection == 0:
            return 0

        TP = 0
        FN = 0

        for l in label:
            iou_score = max(iou(l, pred[i]) for i in range(valid_detection))
            if iou_score > threshold:
                # label bbox detected
                TP += 1
            else:
                # label bbox not detected
                FN += 1

        return float(TP / (TP + FN))

    return np.mean([recall_per_batch(label, pred, valid_detection) for label, pred, valid_detection in
                    zip(y_true, y_pred, valid_detections)])


def mAP(y_true, y_pred, scores, valid_detections, threshold):
    def mAP_per_batch(label, pred, score, valid_detection):
        if valid_detection == 0:
            return 0

        pred = np.asarray(pred[:valid_detection])
        score = np.asarray(score[:valid_detection])

        # sort predictions by score in descending order
        sorted_index = np.argsort(score)[::-1]
        pred = pred[sorted_index]

        label = np.asarray(label)
        GT_class = len(label)

        detected_label = []
        recalls = []
        precisions = []
        ap_r = [0.0 for _ in range(11)]
        TP = FP = 0

        for p in pred:
            iou_scores = np.asarray([iou(l, p) for l in label])
            # set the iou score of the labels which have been detected to 0
            # label
            iou_scores[detected_label] = 0.0
            iou_score, index = np.max(iou_scores), np.argmax(iou_scores)
            if iou_score > threshold:
                TP += 1
                detected_label.append(index)
            else:
                FP += 1

            # update stat
            recall = TP / GT_class
            precision = TP / (TP + FP)
            recalls.append(recall)
            precisions.append(precision)

        # calculate precision at each recall level
        for precision, recall in zip(precisions, recalls):
            recall_index = int(round(recall / 0.1, 2))
            ap_r = [max(value, precision) if i <= recall_index else value for i, value in enumerate(ap_r)]

        return np.mean(ap_r)

    m_aps = [mAP_per_batch(label, pred, score, valid_detection) for label, pred, score, valid_detection in
             zip(y_true, y_pred, scores, valid_detections)]

    return np.mean(m_aps)


def yolo_loss(pred_sbbox, pred_mbbox, pred_lbbox, true_sbbox, true_mbbox, true_lbbox):
    anchors = yolo_anchors
    masks = yolo_anchor_masks
    loss_sbbox = loss_layer(pred_sbbox, true_sbbox, anchors[masks[0]])
    loss_mbbox = loss_layer(pred_mbbox, true_mbbox, anchors[masks[1]])
    loss_lbbox = loss_layer(pred_lbbox, true_lbbox, anchors[masks[2]])

    return tf.reduce_sum(loss_sbbox + loss_mbbox + loss_lbbox)


# reference code from https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/models.py
def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def loss_layer(y_pred, y_true, anchors):
    # 1. transform all pred outputs
    # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
    pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
        y_pred, anchors)
    pred_xy = pred_xywh[..., 0:2]
    pred_wh = pred_xywh[..., 2:4]

    # 2. transform all true outputs
    # y_true: (batch_size, grid, grid, anchors, (x, y, w, h, obj, cls))
    true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
    true_xy = true_box[..., 0:2]
    true_wh = true_box[..., 2:4]

    # give higher weights to small boxes
    box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

    # 3. inverting the pred box equations
    grid_size = tf.shape(y_true)[1]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
        tf.cast(grid, tf.float32)
    true_wh = tf.math.log(true_wh / anchors)
    true_wh = tf.where(tf.math.is_inf(true_wh),
                       tf.zeros_like(true_wh), true_wh)

    # 4. calculate all masks
    obj_mask = tf.squeeze(true_obj, -1)
    # ignore false positive when iou is over threshold
    best_iou, _, _ = tf.map_fn(
        lambda x: (tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
            x[1], tf.cast(x[2], tf.bool))), axis=-1), 0, 0),
        (pred_box, true_box, obj_mask))
    ignore_mask = tf.cast(best_iou < params["yolo_score_threshold"], tf.float32)

    # 5. calculate all losses
    xy_loss = obj_mask * box_loss_scale * \
        tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
    wh_loss = obj_mask * box_loss_scale * \
        tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
    obj_loss = binary_crossentropy(true_obj, pred_obj)
    obj_loss = obj_mask * obj_loss + \
        (1 - obj_mask) * ignore_mask * obj_loss
    # TODO: use binary_crossentropy instead
    class_loss = obj_mask * sparse_categorical_crossentropy(
        true_class_idx, pred_class)

    # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
    xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
    wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
    obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
    class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

    return xy_loss + wh_loss + obj_loss + class_loss


def yolo_boxes(pred, anchors):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, params["NUM_CLASS"]), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box

# https://github.com/zzh8829/yolov3-tf2/blob/c9408234b4a6ad701df1e8ce486857974622b266/yolov3_tf2/models.py#L204
def YOLOv3(input_tensor, num_classes, name='YOLOv3', training=False):
    def yolo_output(inputs, conv, yolo_anchors, anchors, training=False):
        x = conv(inputs, filters=len(yolo_anchors) * (num_classes + 5), kernel_size=1, activation=False, bn=False, training=training)
        x = tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, num_classes + 5))
        return x
    masks = yolo_anchor_masks
    x, x_61, x_36 = darknet53(input_tensor, training=training).outputs

    # 1st layer bbox
    x = conv2d_unit(x, 512, 1, training=training)
    x = conv2d_unit(x, 1024, 3, training=training)
    x = conv2d_unit(x, 512, 1, training=training)
    x = conv2d_unit(x, 1024, 3, training=training)
    x = conv2d_unit(x, 512, 1, training=training)

    sbbox = conv2d_unit(x, 1024, 3, training=training)
    sbbox = yolo_output(sbbox, conv2d_unit, masks[0], len(masks[0]), training=training)

    # upsampling
    x = conv2d_unit(x, 256, 1, training=training)
    x = UpSampling2D()(x)
    x_61 = Concatenate()([x, x_61])

    # 2nd layer bbox
    x_61 = conv2d_unit(x_61, 256, 1, training=training)
    x_61 = conv2d_unit(x_61, 512, 3, training=training)
    x_61 = conv2d_unit(x_61, 256, 1, training=training)
    x_61 = conv2d_unit(x_61, 512, 3, training=training)
    x_61 = conv2d_unit(x_61, 256, 1, training=training)

    mbbox = conv2d_unit(x_61, 512, 3, training=training)
    mbbox = yolo_output(mbbox, conv2d_unit, masks[1], len(masks[1]), training=training)

    x_61 = conv2d_unit(x_61, 128, 1, training=training)
    x_61 = UpSampling2D()(x_61)
    x_36 = Concatenate()([x_61, x_36])

    x_36 = conv2d_unit(x_36, 128, 1, training=training)
    x_36 = conv2d_unit(x_36, 256, 3, training=training)
    x_36 = conv2d_unit(x_36, 128, 1, training=training)
    x_36 = conv2d_unit(x_36, 256, 3, training=training)
    x_36 = conv2d_unit(x_36, 128, 1, training=training)
    x_36 = conv2d_unit(x_36, 256, 3, training=training)

    lbbox = yolo_output(x_36, conv2d_unit, masks[2], len(masks[2]), training=training)

    return Model(inputs=input_tensor, outputs=[sbbox, mbbox, lbbox], name=name)
