import abc
import linecache
import os

import numpy as np
from tensorflow.keras.utils import Sequence

import cv2


class generator(Sequence):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, annotation_file, batch_size, imsize, num_channels, timesteps, label_len, characters, shuffle=True):
        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.lexicon_path = os.path.join(self.data_dir, 'lexicon.txt')
        self.filenames = self.read_annotation_file()
        self.word_labels = self.read_lexicon()
        self.batch_size = batch_size
        self.imsize = imsize
        self.num_channels = num_channels
        self.timesteps = timesteps
        self.label_len = label_len
        self.characters = characters
        self.shuffle = shuffle
        self.num_samples = len(self.filenames)
        self.on_epoch_end()

    def __len__(self):
        '''return # of batches in an epoch'''
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        '''generates one batch of data'''
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.filenames[i] for i in indices]
        return self._generate_data(batch_files)

    def on_epoch_end(self):
        self.indices = np.arrange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def read_annotation_file(self):
        filenames = []
        with open(self.annotation_file, 'r') as f:
            filenames = f.readlines()
        return filenames

    def read_lexicon(self):
        words = []
        with open(self.lexicon_path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                words.append(l)
        return words

    # get image from path
    def load_img(self, img_path):
        # this is only for fail proof but num_channels should be 3
        if self.num_channels == 1:
            return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.imread(img_path)

    def load_img_and_annotation(self, fn):
        img, word = [],[]
        while True:
            fn_split = fn.split()
            word = self.word_labels[int(fn_split[1])]
            impath = os.path.join(self.data_dir, fn_split[0][2:])
            img = self.load_img(impath)
            if (img is not None) and (len(word)<=self.label_len):
                loaded_img = img.shape
                if loaded_img[0]>2 and loaded_img[1]>2:
                    break
            fn = np.random.choice(self.filenames)
        if loaded_img[1]/loaded_img[0]<6.4:
            img = self.pad_img(img, loaded_img)
        else:
            img = self.resize_img(img)
        return img, word

    def pad_img(self, img, loaded_img):
        '''
        img: (w,h)
        loaded_img: (h,w)
        '''
        img_reshape = cv2.resize(img, (int(self.imsize[1]/loaded_img[0]*loaded_img[1]),self.imsize[1]))
        if self.num_channels==1:
            padding = np.zeros((self.imsize[1], self.imsize[0]-int(self.imsize[1]/loaded_img[0]*loaded_img[1])), dtype=np.int32)
        else:
            padding = np.zeros((self.imsize[1], self.imsize[0]-int(self.imsize[1]/loaded_img[0]*loaded_img[1]), self.num_channels), dtype=np.int32)
        img = np.concatenate([img_reshape, padding], axis=1)
        return img

    def resize_img(self, img):
        img = cv2.resize(img, self.imsize, interpolation=cv2.INTER_CUBIC)
        return np.asarray(img)

    def preprocess(self, img):
        if self.num_channels==1:
            img = img.transpose([1,0])
        else:
            img = img.transpose([1,0,2])
        img = np.flip(img,1)
        img = img/255.
        if self.num_channels==1:
            img=img[:,:,np.newaxis]
        return img

    @abc.abstractmethod
    def _generate_data(self, batch_files):
        '''generate batches of data'''
        pass

class traingen(generator):
    def _generate_data(self, batch_files):
        x = np.zeros((self.batch_size, *self.imsize, self.num_channels))
        y = np.zeros((self.batch_size, self.label_len), dtype=np.uint8)

        for i, fn in enumerate(batch_files):
            img, word = self.load_img_and_annotation(fn)
            img = self.preprocess(img)
            x[i]=img
            while len(word)<self.label_len:
                # padding <EOS> like
                word +='-'
            y[i]=[self.characters.find(c) for c in word]
            return [x,y, np.ones(self.batch_size)*int(self.timesteps-2), np.ones(self.batch_size*self.label_len)], y

class valgen(generator):
    def _generate_data(self, batch_files):
        x = np.zeros((self.batch_size, *self.imsize, self.num_channels))
        y = []

        for i, fn in enumerate(batch_files):
            img, word = self.load_img_and_annotation(fn)
            img = self.preprocess(img)
            x[i]=img
            y.append(word)
        return x,y
