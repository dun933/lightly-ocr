import string
import os

import numpy as np
import cv2
import yaml

working_path = os.path.dirname(os.path.realpath(__file__))
char_dict = string.digits+string.ascii_lowercase+string.ascii_uppercase
with open(os.path.join(working_path,'config.yml'),'r') as s:
    params = yaml.safe_load(s)
    assert params['NUM_CLASSES']==len(char_dict), f'NUM_CLASSES should be the same with len(char_dict). got {params["NUM_CLASSES"]}instead'

def decodetext(char_dict, outputs):
    return ''.join((char_dict[i] for i in outputs))

def sparse_tuple_from(sequences):
    indices, values = list(), list()
    for i, seq in enumerate(sequences):
        indices.extend(zip([i]*len(seq),range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    dense_shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1],dtype=np.int64)

    return indices, values, dense_shape

def preprocess(image, height=params['INPUT_SIZE'][1],width=params['INPUT_SIZE'][0]):
    scale = height/image.shape[0]
    tmp_w = int(scale*image.shape[1])
    nw = width if tmp_w>width else tmp_w
    image = cv2.resize(image, (nw, height), interpolation=cv2.INTER_LINEAR)

    r, c = np.shape(image)
    if c>width:
        ratio =float(width)/c
        image = cv2.resize(image,(width, int(32*ratio)))
    else:
        w_pad = width-image.shape[1]
        image = np.pad(image, pad_width=[(0,0),(0,w_pad)], mode='constant', constant_values=0)

    image = image[:,:,np.newaxis]

    return image

def datagen(char_dict=char_dict, dataset='train',
            batches=params['NUM_BATCHES'],
            batch_size=params['BATCH_SIZE'],
            epochs=params['EPOCHS'],
            data_path=params['DATA_PATH']):
    x_batch, y_batch = list(),list()
    for _ in range(epochs):
        with open(data_path+f'annotation_{dataset}.txt') as f:
            for _ in range(batches*batch_size):
                impath = f.readline().replace('\n','').split(' ')[0]
                image = cv2.imread(data_path+impath.replace('./',''),0)
                if image is None:
                    continue
                x = preprocess(image=image)

                y = impath.split('_')[1]
                y = [char_dict.index(i) if i in char_dict else len(char_dict)-1 for i in y]
                y = y

                x_batch.append(x)
                y_batch.append(y)

                if len(y_batch)==batch_size:
                    yield np.array(x_batch).astype(np.float32), np.array(y_batch)
                    x_batch = []
                    y_batch = []
