import os

import cv2
import imgaug
import numpy as np
from shapely import geometry
from scipy import spatial


def read(fpath):
    '''read file into image object'''
    if isinstance(fpath, np.ndarray):
        return fpath
    if hasattr(fpath, 'read'):
        image = np.asarray(bytearray(fpath.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    elif isinstance(fpath, str):
        assert os.path.isfile(fpath), f'Could not find image at path: {fpath}'
        image = cv2.imread(fpath)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def augment(boxes,
            augmenter: imgaug.augmenters.meta.Augmenter,
            image=None,
            boxes_format='boxes',
            image_shape=None,
            area_threshold=0.5,
            min_area=None):
    '''augment an image and associated boxes together.'''
    if image is None and image_shape is None:
        raise ValueError('One of "image" or "image_shape" must be provided.')
    augmenter = augmenter.to_deterministic()

    if image is not None:
        image_augmented = augmenter(image=image)
        image_shape = image.shape[:2]
        image_augmented_shape = image_augmented.shape[:2]
    else:
        image_augmented = None
        width_augmented, height_augmented = augmenter.augment_keypoints(
            imgaug.KeypointsOnImage.from_xy_array(xy=[[image_shape[1], image_shape[0]]],
                                                  shape=image_shape)).to_xy_array()[0]
        image_augmented_shape = (height_augmented, width_augmented)

    def box_inside_image(box):
        area_before = cv2.contourArea(np.int32(box)[:, np.newaxis, :])
        if area_before == 0:
            return False, box
        clipped = box.copy()
        clipped[:, 0] = clipped[:, 0].clip(0, image_augmented_shape[1])
        clipped[:, 1] = clipped[:, 1].clip(0, image_augmented_shape[0])
        area_after = cv2.contourArea(np.int32(clipped)[:, np.newaxis, :])
        return ((area_after / area_before) >= area_threshold) and (min_area is None or
                                                                   area_after > min_area), clipped

    def augment_box(box):
        return augmenter.augment_keypoints(
            imgaug.KeypointsOnImage.from_xy_array(box, shape=image_shape)).to_xy_array()

    if boxes_format == 'boxes':
        boxes_augmented = [
            box for inside, box in [box_inside_image(box) for box in map(augment_box, boxes)]
            if inside
        ]
    elif boxes_format == 'lines':
        boxes_augmented = [[(augment_box(box), character) for box, character in line]
                           for line in boxes]
        boxes_augmented = [[(box, character)
                            for (inside, box), character in [(box_inside_image(box), character)
                                                             for box, character in line] if inside]
                           for line in boxes_augmented]
        # Sometimes all the characters in a line are removed.
        boxes_augmented = [line for line in boxes_augmented if line]
    elif boxes_format == 'predictions':
        boxes_augmented = [(word, augment_box(box)) for word, box in boxes]
        boxes_augmented = [(word, box) for word, (inside, box) in [(word, box_inside_image(box))
                                                                   for word, box in boxes_augmented]
                           if inside]
    else:
        raise NotImplementedError(f'Unsupported boxes format: {boxes_format}')
    return image_augmented, boxes_augmented


def resize_image(image, max_scale, max_size):
    '''Obtain the optimal resized image subject to a maximum scaleand maximum size.'''
    if max(image.shape) * max_scale > max_size:
        # We are constrained by the maximum size
        scale = max_size / max(image.shape)
    else:
        # We are contrained by scale
        scale = max_scale
    return cv2.resize(image,
                      dsize=(int(image.shape[1] * scale), int(image.shape[0] * scale))), scale


def fit(img, w, h, cval=255, mode='letterbox', return_scale=False):
    '''literally another way to resize img to desired output'''
    fitted = None
    xs = w / img.shape[1]
    ys = w / img.shape[0]
    if xs == 1 and ys == 1:
        fitted = img
        scale = 1
    elif (xs <= ys and mode == 'letterbox') or (xs >= ys and mode == 'crop'):
        scale = xs
        rw = w
        rh = (w / img.shape[1]) * img.shape[0]
    else:
        scale = ys
        # get new resized height and width
        rh = h
        rw = scale * img.shape[1]
    if fitted is None:
        rws, rhs = map(int, [rw, rh])
        if mode == 'letterbox':
            fitted = np.zeros((h, w, 3), dtype='uint8') + cval
            img = cv2.resize(img, dsize=(rws, rhs))
            fitted[:img.shape[0], :img.shape[1]] = img[:h, :w]
        elif mode == 'crop':
            img = cv2.resize(img, dsize=(rws, rhs))
            fitted = img[:h, :w]
        else:
            raise NotImplementedError(f'unsupported mode: {mode}')
    if not return_scale:
        return fitted
    return fitted, scale


def read_fit(fpath, width,height, cval, mode='letterbox'):
    img = read(fpath) if isinstance(fpath, str) else fpath
    img = fit(img=img, w=width, h=height, cval=cval, mode=mode)
    return img


def adjust_boxes(boxes, boxes_format='boxes', scale=1):
    '''adjust boxes given scale'''
    if scale == 1:
        return boxes
    if boxes_format == 'boxes':
        return np.array(boxes) * scale
    if boxes_format == 'lines':
        return [[(np.array(box) * scale, character) for box, character in line] for line in boxes]
    if boxes_format == 'predictions':
        return [(word, np.array(box) * scale) for word, box in boxes]
    raise NotImplementedError(f'Unsupported boxes format: {boxes_format}')


def warpbox(image,
            box,
            target_height=None,
            target_width=None,
            margin=0,
            cval=None,
            return_transform=False):
    '''warp a boxed region given by set of four points into a rectangle'''
    if cval is None:
        cval = (0, 0, 0) if len(image.shape) == 3 else 0

    def get_rotated_width_height(box):
        '''returns the width and height of a rotated rectangle'''
        w = (spatial.distance.cdist(box[0][np.newaxis], box[1][np.newaxis], "euclidean") +
             spatial.distance.cdist(box[2][np.newaxis], box[3][np.newaxis], "euclidean")) / 2
        h = (spatial.distance.cdist(box[0][np.newaxis], box[3][np.newaxis], "euclidean") +
             spatial.distance.cdist(box[1][np.newaxis], box[2][np.newaxis], "euclidean")) / 2
        return int(w[0][0]), int(h[0][0])

    def get_rotated_box(points):
        '''Obtain the parameters of a rotated box.'''
        try:
            mp = geometry.MultiPoint(points=points)
            pts = np.array(list(zip(*mp.minimum_rotated_rectangle.exterior.xy)))[:-1]  # noqa: E501
        except AttributeError:
            # There weren't enough points for the minimum rotated rectangle function
            pts = points
        # The code below is taken from
        # https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py

        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        pts = np.array([tl, tr, br, bl], dtype="float32")

        rotation = np.arctan((tl[0] - bl[0]) / (tl[1] - bl[1]))
        return pts, rotation

    box, _ = get_rotated_box(box)
    w, h = get_rotated_width_height(box)
    assert (
        (target_width is None and target_height is None)
        or (target_width is not None and target_height is not None)), \
            'Either both or neither of target width and height must be provided.'
    if target_width is None and target_height is None:
        target_width = w
        target_height = h
    scale = min(target_width / w, target_height / h)
    M = cv2.getPerspectiveTransform(src=box,
                                    dst=np.array([[margin, margin], [scale * w - margin, margin],
                                                  [scale * w - margin, scale * h - margin],
                                                  [margin, scale * h - margin]]).astype('float32'))
    crop = cv2.warpPerspective(image, M, dsize=(int(scale * w), int(scale * h)))
    target_shape = (target_height, target_width, 3) if len(image.shape) == 3 else (target_height,
                                                                                   target_width)
    full = (np.zeros(target_shape) + cval).astype('uint8')
    full[:crop.shape[0], :crop.shape[1]] = crop
    if return_transform:
        return full, M
    return full

