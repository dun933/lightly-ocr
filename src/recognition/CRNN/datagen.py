import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms

import lmdb
import six
import sys
from PIL import Image
import numpy as np

class LMDBDataset(Dataset):
    def __init__(self,root=None, transform=None, target_transform=None):
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print(f'cannot create lmdb from {root}')
            sys.exit(0)
        with self.env.begin(write=False) as db:
            num_samples = int(db.get('num_samples'))
            self.num_samples = num_samples

        self.transform=transform
        self.target_transform=target_transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        assert index <=len(self), 'index range error'
        index +=1
        with self.env.begin(write=False) as db:
            imgkey = f'image-{index}'
            imgbuf = db.get(imgkey)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print(f'corrupted image for {index}')
                return self[index+1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = f'label-{index}'
            label = str(db.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)

class resize_normalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size=size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class random_sampler(sampler.Sampler):
    def __init__(self, data_dir, batch_size):
        self.num_samples = len(data_dir)
        self.batch_size = batch_size

    def __iter__(self):
        num_batch = len(self)//self.batch_size
        tail = len(self)%self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(num_batch):
            random_start = random.randint(0, len(self)-self.batch_size)
            batch_idx = random_start + torch.range(0, self.batch_size - 1)
            index[i*self.batch_size:(i+1)*self.batch_size] = batch_idx
        if tail:
            random_start = random.randint(0, len(self)-self.batch_size)
            tail_idx = random_start+torch.range(0, tail-1)
            index[(i+1)*self.batch_size:] = tail_idx
        return iter(index)

    def __len__(self):
        return self.num_samples

class align_collate(object):
    def __init__(self, height=32, width=100, keep_ratio=False, min_ratio=1):
        self.height = height
        self.width = width
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        h,w = self.height, self.width
        if self.keep_ratio:
            ratios = []
            for img in images:
                w,h = img.size
                ratios.append(w/float(h))
            ratios.sort()
            max_ratio = ratio[-1]
            w = int(np.floor(max_ratio*h))
            w = max(h*self.min_ratio, w) # -> make sure h>w

        transform - resize_normalize((w,h))
        images = [transform(img) for img in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
