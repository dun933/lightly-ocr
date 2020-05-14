import os
import sys
import re
import six
import math

import lmdb
import torch

from natsort import natsorted
from PIL import Image
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, Subset

def tensor2im(image_tensor, imtype=np.uint8):
    im_np = image_tensor.cpu().float().numpy()
    if im_np.shape[0] == 1:
        im_np = np.tile(im_np, (3, 1, 1))
    im_np = (np.transpose(im_np, (1, 2, 0)) + 1) / 2. * 255.
    return im_np.astype(imtype)

def save_image(image_numpy, impath):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(impath)

# taken from python docs
def _accumulate(iterable, fn=lambda x, y: x + y):
    # return running totals
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

class resize_normalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class normalize_pad(object):
    def __init__(self, max_size, pad_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.pad_type = pad_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        padded = torch.FloatTensor(*self.max_size).fill_(0)
        padded[:, :, w:] = img # -> right pad
        if self.max_size[2] != w:
            padded[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return padded

class align_collate(object):
    def __init__(self, height=32, width=100, keep_ratio_with_pad=False):
        self.height = height
        self.width = width
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:
            resized_max_w = self.width
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = normalize_pad((input_channel, self.height, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.height * ratio) > self.width:
                    resized_w = self.width
                else:
                    resized_w = math.ceil(self.height * ratio)

                resized = image.resize((resized_w, self.height), Image.BICUBIC)
                resized_images.append(transform(resized))

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        else:
            transform = resize_normalize((self.width, self.height))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels

class BalancedDataset(object):
    # 50-50 for synthtext and mjsynth
    def __init__(self, config):
        dashed = '-' * 80
        log = open(f'{config["log_dir"]}/log_dataset.txt', 'a')
        log.write(dashed + '\n')
        log.write(f'train data:{config["data_dir"]}\nselected:{config["select_data"]}\nbatch_ratio:{config["batch_ratio"]}\n')

        _align_collate = align_collate(height=config['height'], width=config['width'], keep_ratio_with_pad=config['pad'])
        self.generator_list, self.generator_iter_list = [], []
        batch_size_list = []
        total_batch_size = 0

        for selected_, batch_ratio_ in zip(config['select_data'], config['batch_ratio']):
            _batch_size = max(round(config['batch_size'] * float(batch_ratio_)), 1)
            log.write(dashed + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=config['train_data'], config=config, select_data=[selected_])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

        # total number of data can be modified with usage_ratio
            number_dataset = int(total_number_dataset * float(config['usage_ratio']))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_log = f'total samples: {selected_}: {total_number_dataset} x {config["usage_ratio"]}(usage_ratio)={len(_dataset)}\n'
            selected_log += f'num_samples per batch: {config["batch_size"]} x {float(batch_ratio_)}(batch_ratio) = {_batch_size}'
            log.write(selected_log + '\n')
            batch_size_list.append(str(_batch_size))
            total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(_dataset, batch_size=_batch_size, shuffle=True, num_workers=int(config['workers']),
                                                       collate_fn=_align_collate, pin_memory=True)
            self.generator_list.append(_data_loader)
            self.generator_iter_list.append(iter(_data_loader))

        total_batch_size_log = f'{dashed}\n'
        sum_batch_size = '+'.join(batch_size_list)
        total_batch_size_log += f'total batch size: {sum_batch_size} = {total_batch_size}\n{dashed}'
        config['batch_size'] = total_batch_size
        log.write(total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images, balanced_batch_texts = [], []

        for i, dataloader_iter in enumerate(self.generator_iter_list):
            try:
                image, text = dataloader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.generator_iter_list[i] = iter(self.generator_list[i])
                image, text = self.generator_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

            balanced_batch_images = torch.cat(balanced_batch_images, 0)
            return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, config, select_data='/'):
    # select data returns all subdir
    dataset_list = []
    dataset_log = f'dataset_root: {root}\t dataset: {select_data[0]}\n'
    for dirpath, dirnames, filenames in os.walk(root + '/'):
        if not dirnames:
            select_flag = False
            for selected_ in select_data:
                if selected_ in dirpath:
                    select_flag = True
                    break
            if select_flag:
                dataset = LMDBDataset(dirpath, config)
                sub_dataset_log = f'subdir: /{os.path.relpath(dirpath, root)}\nnum_samples: {len(dataset)}'
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset, dataset_log

class LMDBDataset(Dataset):
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print(f'cannot create LMDB from {root}')
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            self.num_samples = num_samples

            if self.config['data_filtering_off']:
                self.filtered_index_list = [index + 1 for index in range(self.num_samples)]
            else:
                self.filtered_index_list = []
                for index in range(self.num_samples):
                    index += 1
                    label_key = f'label-{index}'.encode()
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.config['batch_max_len']:
                        continue
                    out_of_char = f'[^{self.config["character"]}]'
                    if re.search(out_of_char, label.lower()):
                        continue
                    self.filtered_index_list.append(index)

                self.num_samples = len(self.filtered_index_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = f'label-{index}'.encode()
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-{index}'.encode()
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.config[' rgb ']:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.config[' rgb ']:
                    img = Image.new('RGB', (self.config['width'], self.config['height']))
                else:
                    img = Image.new('L', (self.config['width'], self.config['height']))
                label = '[dummy_label]'

            if not self.config['sensitive']:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.config["character"]}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)
