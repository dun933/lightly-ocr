import os
import cv2
import glob
import warnings
import shutil
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

def create_result_subdir(result_dir):
    while True:
        rid = 0
        for fname in glob.glob(os.path.join(result_dir, '*')):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase)
                rid = max(rid, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, '%03d'% (rid))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    return result_subdir

def create_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_subdir = create_result_subdir(output_dir)
    print('Output directory: ' + output_subdir)
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        shutil.copy(f, output_subdir)
    return output_subdir

class PredictionCheckpoint(Callback):
    def __init__(self, fpath, pred_model, monitor='loss', save_best_only=False, mode='min', period=1, save_weights_only=False, verbose=False):
        self.fpath = fpath
        self.pred_model = pred_model
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.period = period
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.epochs_since_last_save = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.fpath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(f'Can save best model only with {self.monitor} available, '
                                  'skipping.', RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(f'\nEpoch {epoch+1}: {self.monitor} improved from {self.best} to {current}, '
                                  f' saving model to {filepath}')
                        self.best = current
                        if self.save_weights_only:
                            self.pred_model.save_weights(filepath, overwrite=True)
                        else:
                            self.pred_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print(f'\nEpoch {epoch+1}: {self.monitor} did not improve from {self.best}' )
            else:
                if self.verbose > 0:
                    print(f'\nEpoch {epoch+1}: saving model to {filepath}')
                if self.save_weights_only:
                    self.pred_model.save_weights(filepath, overwrite=True)
                else:
                    self.pred_model.save(filepath, overwrite=True)


class Evaluator(Callback):

    def __init__(self, prediction_model, val_generator, label_len, characters, optimizer, period=2000):
        self.pred_model = prediction_model
        self.period = period
        self.val_generator = val_generator
        self.label_len = label_len
        self.characters = characters
        self.optimizer = optimizer

    def on_batch_end(self, batch, logs=None):
        if ((batch + 1) % self.period) == 0:
            accuracy, correct_char_predictions = self.evaluate()
            print('=====================================')
            print(f'Word level accuracy: {accuracy}')
            print(f'Correct character level predictions: {correct_char_predictions}')
            print('=====================================')

    def on_epoch_end(self, epoch, logs=None):
        accuracy, correct_char_predictions = self.evaluate()
        print('=====================================')
        print(f'After epoch {epoch}')
        print(f'Word level accuracy: {accuracy}')
        print(f'Correct character level predictions: {correct_char_predictions}')
        if self.optimizer == 'sgd':
            lr = self.model.optimizer.lr
            decay = self.model.optimizer.decay
            iterations = self.model.optimizer.iterations
            lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
            print(f"Decayed learning rate: {K.eval(lr_with_decay)}")
        else:
            print(f"Learning rate: {K.eval(self.model.optimizer.lr)}")

    def evaluate(self):
        correct_predictions = 0
        correct_char_predictions = 0

        x_val, y_val = self.val_generator[np.random.randint(0, int(self.val_generator.nb_samples / self.val_generator.batch_size))]
        #x_val, y_val = next(self.val_generator)

        y_pred = self.pred_model.predict(x_val)

        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
        ctc_out = K.get_value(ctc_decode)[:, :self.label_len]

        for i in range(self.val_generator.batch_size):
            print(ctc_out[i])
            result_str = ''.join([self.characters[c] for c in ctc_out[i]])
            result_str = result_str.replace('-', '')
            if result_str == y_val[i]:
                correct_predictions += 1
            print(result_str, y_val[i])

            for c1, c2 in zip(result_str, y_val[i]):
                if c1 == c2:
                    correct_char_predictions += 1

        return correct_predictions / self.val_generator.batch_size, correct_char_predictions


def pad_img(img, imsize, nb_channels):
    # imsize : (width, height)
    # loaded_img : (height, width)
    img_reshape = cv2.resize(img, (int(imsize[1] / img.shape[0] * img.shape[1]), imsize[1]))
    if nb_channels == 1:
        padding = np.zeros((imsize[1], imsize[0] - int(imsize[1] / img.shape[0] * img.shape[1])), dtype=np.int32)
    else:
        padding = np.zeros((imsize[1], imsize[0] - int(imsize[1] / img.shape[0] * img.shape[1]), nb_channels), dtype=np.int32)
    img = np.concatenate([img_reshape, padding], axis=1)
    return img

def resize_img(img, imsize):
    img = cv2.resize(img, imsize, interpolation=cv2.INTER_CUBIC)
    img = np.asarray(img)
    return img
