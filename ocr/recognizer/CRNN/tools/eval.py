import os
import re
import string
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from nltk.metrics.distance import edit_distance

from dataset import align_collate, hierarchical_dataset
from utils import AttnLabelConverter, Averager, CTCLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validation(model, loss_fn, eval_loader, converter, config):
    # loss_fn: torch.nn.CTCLoss(zero_infinity=True) or torch.nn.CrossEntropyLoss(ignore_index=0)
    # converter: either AttnLabelConverter, CTCLabelConverter
    # config: config.yml, parsed with pyyaml
    num_correct = 0
    norm_ED = 0
    len_data = 0
    infer_time = 0
    avg_valid_loss = Averager()

    for i, (image_tensor, labels) in enumerate(eval_loader):
        batch_size = image_tensor.size(0)
        len_data += batch_size
        image = image_tensor.to(device)
        len_pred = torch.IntTensor(config['batch_max_len'] *
                                   batch_size).to(device)
        text_pred = torch.LongTensor(batch_size, config['batch_max_len'] +
                                     1).fill_(0).to(device)

        text_loss, len_loss = converter.encode(
            labels, batch_max_len=config['batch_max_len'])

        start = time.time()
        if config['prediction'] == 'CTC':
            preds = model(image, text_pred)
            forward = time.time() - start

            # get eval loss with CTC
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            cost = loss_fn(
                preds.log_softmax(2).permute(1, 0, 2), text_loss, preds_size,
                len_loss)

            _, preds_idx = preds.max(2)
            preds_idx = preds_idx.view(-1)
            preds_str = converter.decode(preds_idx.data, preds_size.data)

        else:
            preds = model(image, text_pred, training=False)
            forward = time.time() - start
            preds = preds[:, :text_loss.shape[1] - 1, :]
            target = text_loss[:, 1:]  # remove [GO] token
            cost = loss_fn(preds.contiguous().view(-1, preds.shape[1]),
                           target.contiguous().view(-1))

            # establish greedy decoder then decode idx to char
            _, preds_idx = preds.max(2)
            preds_str = converter.decode(preds_idx, len_pred)
            labels = converter.decode(text_loss[:, 1:], len_loss)

        infer_time += forward
        avg_valid_loss.add(cost)

        # returns accuracy and confidence score, cofidence score is useless tho
        probs = F.softmax(preds, dim=2)
        max_probs = probs.max(dim=2)
        confidence_ = []
        for gt, pred, max_prob in zip(labels, preds_str, max_probs):
            if config['prediction'] == 'Attention':
                gt = gt[:gt.find('[s]')]  # returns ground gruth EOS
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # remove EOS token [s]
                max_prob = max_prob[:pred_EOS]

            if config['sensitive'] and config['filtering']:
                pred, gt = pred.lower(), gt.lower()
                case_sensitive = string.digits + string.ascii_lowercase
                except_case = f'[^{case_sensitive}]'
                pred, gt = [re.sub(except_case, '', i) for i in [pred, gt]]

            if pred == gt:
                num_correct += 1
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)
            try:
                score = max_prob.cumprod(dim=0)[-1]
            except:
                score = 0  # -> empty case return 0 after pruning EOS token
            confidence_.append(score)
    accuracy = num_correct / float(len_data) * 100
    norm_ED = norm_ED / float(len_data)
    valid_loss = avg_valid_loss.val()

    return valid_loss, accuracy, norm_ED, preds_str, confidence_, labels, infer_time, len_data
