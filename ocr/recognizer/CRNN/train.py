import os
import random
import string
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from model import CRNN
from tools.dataset import LMDBDataset, align_collate, random_sequential_sampler
from tools.utils import AttnLabelConverter, Averager, CTCLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     'config.yml'), 'r') as f:
    CONFIG = yaml.safe_load(f)
DASHED = '-' * 80
random.seed(CONFIG['seeds'])
np.random.seed(CONFIG['seeds'])
torch.manual_seed(CONFIG['seeds'])
torch.cuda.manual_seed(CONFIG['seeds'])
cudnn.benchmark = True
cudnn.deterministic = True

train_dataset = LMDBDataset(CONFIG['train_root'])
if not CONFIG['random_sample']:
    sampler = random_sequential_sampler(train_dataset, CONFIG['batch_size'])
else:
    sampler = None
collate_fn = align_collate(height=CONFIG['height'],
                           width=CONFIG['width'],
                           keep_ratio_with_pad=CONFIG['keep_ratio'])
train_loader = DataLoader(train_dataset,
                          batch_size=CONFIG['batch_size'],
                          shuffle=True,
                          sampler=sampler,
                          num_workers=int(CONFIG['workers']),
                          collate_fn=collate_fn,
                          pin_memory=True)
CONFIG['num_classes'] = len(CONFIG['character']) + 1
if CONFIG['prediction'] == 'CTC':
    converter = CTCLabelConverter(CONFIG['character'])
else:
    converter = AttnLabelConverter(CONFIG['character'])
CONFIG['num_classes'] = len(converter.character)

if CONFIG['rgb']:
    CONFIG['input_channel'] = 3
model = CRNN(CONFIG)
print(
    f'model input params:\nheight:{CONFIG["height"]}\nwidth:{CONFIG["width"]}\nfidicial points:{CONFIG["num_fiducial"]}\ninput channel:{CONFIG["input_channel"]}\noutput channel:{CONFIG["output_channel"]}\nhidden size:{CONFIG["hidden_size"]}\nnum class:{CONFIG["num_classes"]}\nbatch_max_len:{CONFIG["batch_max_len"]}\nmodel structures as follow:{CONFIG["transform"]}-{CONFIG["backbone"]}-{CONFIG["sequence"]}-{CONFIG["prediction"]}'
)

for name, params in model.named_parameters():
    if 'loc_fc2' in name:
        print(f'skips {name} since fc2 is already initialized')
        continue
    try:
        if 'bias' in name:
            init.constant_(params, 0.)
        elif 'weight' in name:
            init.kaiming_normal_(params)  # torch way of saying he_norm
    except Exception as e:  # for batchnorm
        print(e)
        if 'weight' in name:
            params.data.fill_(1)
        continue
# if you have multiple gpu go ahead I only have 1060 =(
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model).to(device)
model.train()
if CONFIG['saved_model_path'] != '':
    print(f'loading pretrained models from {CONFIG["saved_model_path"]}')
    if CONFIG['fine_tune']:
        model.load_state_dict(torch.load(CONFIG['saved_model_path']),
                              strict=False)
    else:
        model.load_state_dict(torch.load(CONFIG['saved_model_path']))
print(f'Model:{model}')

# get loss
if CONFIG['prediction'] == 'CTC':
    loss_fn = torch.nn.CTCLoss(zero_infinity=True).to(device)
else:
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0).to(
        device)  # [GO] idx at 0
avg_loss = Averager()

filtered_params, num_params = [], []
for p in filter(lambda p: p.requires_grad, model.parameters()):
    filtered_params.append(p)
    num_params.append(np.prod(p.size()))
print(f'trainable params: {sum(num_params)}')

# get optimizer
if CONFIG['adam']:
    optimizer = optim.Adam(filtered_params,
                           lr=CONFIG['lr'],
                           betas=(CONFIG['beta1'], 0.999))
else:
    optimizer = optim.Adadelta(filtered_params,
                               lr=CONFIG['lr'],
                               rho=CONFIG['rho'],
                               eps=CONFIG['eps'])
print(f'optimizer: {optimizer}')


def eval(net, dataset, loss_fn, max_iter=CONFIG['val_interval']):
    print('start val')


def train(CONFIG):

    # training starts here
    start_iter = 0
    if CONFIG['saved_model_path'] != '':
        try:
            start_iter = int(
                CONFIG['saved_model_path'].split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start = time.time()
    best_acc = -1
    best_norm_ED = -1
    i = start_iter

    while True:
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels,
                                        batch_max_len=CONFIG['batch_max_len'])
        batch_size = image.size(0)

        if CONFIG['prediction'] == 'CTC':
            preds = model(image, text).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)

            # disable cudnn for ctc_loss
            torch.backends.cudnn.enabled = False
            cost = loss_fn(preds, text.to(device), preds_size.to(device),
                           length.to(device))
            torch.backends.cudnn.enabled = True

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # prune [GO] symbols
            cost = loss_fn(preds.view(-1, preds.shape[-1]),
                           target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()

        avg_loss.add(cost)

        # val
        if i % CONFIG['val_interval'] == 0:
            elapsed = time.time() - start
            with open(os.path.join(CONFIG['log_dir'], 'log_train.txt'),
                      'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, cur_acc, cur_norm_ED, preds, confidence_, labels, infer_time, len_data = validation(
                        model, loss_fn, loader, converter, CONFIG)
                model.train()

                loss_log = f'[{i}/{CONFIG["num_iters"]}] train loss: {avg_loss.val():0.5f} | valid loss: {valid_loss} | elapsed: {elapsed:0.5f}'
                avg_loss.reset()

                model_log = f'{"current_accuracy":17s}:{cur_acc:0.3f} | {"current_norm_ED":17s}: {cur_norm_ED:0.2f}'

                if cur_acc > best_acc:
                    best_acc = cur_acc
                    torch.save(
                        model.state_dict(),
                        f'{os.path.join(CONFIG["saved_model_path"], "best_acc.pth")}'
                    )
                if cur_norm_ED > best_norm_ED:
                    best_norm_ED = cur_norm_ED
                    torch.save(
                        model.state_dict(),
                        f'{os.path.join(CONFIG["saved_model_path"], "best_norm_ED.pth")}'
                    )
                best_model_log = f'{"best_accuracy":17s}: {best_acc:0.3f} | {"best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # get prediction
                head = f'{"ground truth":25s} | {"prediction"}:25s | confidence score & preds vs. gt'
                pred_logs = f'{DASHED}\n{head}\n{DASHED}\n'
                for gt, pred, confidence in zip(labels[5:], preds[5:],
                                                confidence_[:5]):
                    if CONFIG['prediction'] == 'Attention':
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    pred_logs += f'{gt:25s} | {preds:25s} | {confidence:0.4f}\t{str(pred==gt)}\n'
                pred_logs += f'{DASHED}'
                print(pred_logs)
                log.write(pred_logs + '\n')

        if (i + 1) & 1e+5 == 0:
            torch.save(
                model.state_dict(),
                f'{os.path.join(CONFIG["log_dir"], f"iter_{i+1}.pth")}')

        if i == CONFIG['num_iters']:
            print('stopped training.')
            sys.exit(0)
        i += 1
