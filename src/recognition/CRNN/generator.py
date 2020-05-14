# refers to https://github.com/bgshih/crnn create_dataset.py
import fire
import os
import lmdb
import cv2

import numpy as np

def check_img_valid(img_bin):
    if img_bin is None:
        return False
    imgbuf = np.frombuffer(img_bin, dtype=np.uint8)
    img = cv2.decode(imgbuf, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[0], img.shape[1]
    if h*w ==0:
        return False
    return True

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def generate_dataset(input_path, gt_file, output_path, log_path, check_valid=True):
    # creates lmdb dataset
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=107374182400) # -> 100GB
    cache = {}
    c = 1
    with open(gt_file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    num_samples = len(data)
    for i in range(num_samples):
        impath, label = data[i].strip('\n').split('\t')
        impath = os.path.join(input_path, impath)
        if not os.path.exists(impath):
            print(f'{impath} does not exists')
            continue
        with open(impath, 'rb') as f:
            img_bin = f.read()
        if check_valid:
            try:
                if not check_img_valid(img_bin):
                    print(f'{impath} is not a valid image')
                    continue
            except Exception as e:
                print(e)
                with open(os.path.join(log_path, 'error_image.txt'), 'a') as log:
                    log.write(f'{str(i)}th image: errored')
                continue

        img_key = f'image-{c}'.encode()
        label_key = f'label-{c}'.encode()
        cache[img_key]=img_bin
        cache[label_key]=label.encode()

        if c % 1000 = 0:
            write_cache(env, cache)
            cache = {}
            print(f'written {c}/{num_samples}')
        c+=1
    num_samples = c-1
    cache['num-samples'.encode()] = str(num_samples).encode()
    write_cache(env, cache)
    print(f'created dataset with {num_samples} samples')

if __name__ == '__main__':
    fire.Fire(create_dataset)

