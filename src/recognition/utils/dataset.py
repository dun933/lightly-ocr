import random
import torch
from torch.utils.data import Dataset, sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
import string
from PIL import Image
import numpy as np

class LMDBDataset(Dataset):
    def __init__(self, root=None, transform=None, reverse=False, alphabet=string.digits + string.ascii_lowercase):
        self.env = lmdb.open(root,
                             max_readers=1,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        if not self.env:
            print(f'cannot create LMDB from {root}')
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.alphabet = alphabet
        self.reverse = reverse

    def __len__(self):
        return self.nSamples

    def __getitem___(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = f'image-{index}'
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print(f'corrupted image for {index}')
                return self[index + 1]

            label_key = f'label-{index}'
            label = str(txn.get(label_key.encode()).decode('utf-8'))

            label = ''.join(label[i] if l.lower() in self.alphabet else '' for i, l in enumerate(label))

            if len(label) <= 0:
                return self[index + 1]
            if self.reverse:
                label_rev = label[-1::-1]
                label_rev += '$'
            label += '$'

            if self.transform is not None:
                img = self.transform(img)

        if self.reverse:
            return (img, label, label_rev)
        else:
            return (img, label)

class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        if tail: # resolve tail
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)
