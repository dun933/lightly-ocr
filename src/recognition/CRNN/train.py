import datetime
import os

import numpy as np
import tensorflow as tf

from model import CRNN
from utils import *

os.makedirs('logs', exist_ok=True)
model_path = os.path.join(params['MODEL_PATH'], 'crnn')
os.makedirs(model_path, exist_ok=True)
model = CRNN(num_classes=params['NUM_CLASSES'], training=params['training'])
_ = [model.load_weights(os.path.join(model_path, 'def_weights.h5')) if params['continue'] else True]

# training
# since mjsynth has 7224612 images for training -> 112884 batches of batch_size=64
iters = params['iter']
loss_ = []
loss_train = []
loss_test = []
accuracy = []
curr_accuracy = 0
total_case = 0
total_case_train = 0
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join('logs', 'gradient_tape', current_time, 'train')
test_log_dir = os.path.join('logs', 'gradient_tape', current_time, 'test')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

curr_epoch = 0

aopt = tf.keras.optimizers.Adam(learning_rate=params['lr'], clipnorm=5)
for x_batch, y_batch in datagen():
    # create training ops
    indices, values, dense_shape = sparse_tuple_from(y_batch)
    y_batch_sparse = tf.sparse.SparseTensor(
        indices=indices, values=values, dense_shape=dense_shape)

    with tf.GradientTape() as tape:
        logits, pred, outputs = model(x_batch, training=params['training'])
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y_batch_sparse,
                                             logits=outputs,
                                             label_length=[len(i)for i in y_batch],
                                             logit_length=[params['SEQ_LENGTH']] * len(y_batch),
                                             blank_index=params['NUM_CLASSES']))
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(model.trainable_variables, grads)]

    aopt.apply_gradients(zip(grads, model.trainable_variables))
    print(iters, loss)
    decoded_, _ = tf.nn.ctc_greedy_decoder(outputs,
                                           sequence_length=[params['SEQ_LENGTH']] * len(y_batch),
                                           merge_repeated=True)

    decoded_ = tf.sparse.to_dense(decoded_[0]).numpy()
    print(f"decoded from batch: {[decodetext(char_dict,[char for char in np.trim_zeros(np.array(word), 'b')]) for word in (y_batch)[:4]]}")
    print(f"decoded from ctc greedy: {[decodetext(char_dict,[char for char in np.trim_zeros(word, 'b')]) for word in decoded_[:4]]}")

    train_loss = loss.numpy().round(1)

    loss_.append(loss.numpy().round(1))
    loss_train.append(loss.numpy().round(1))

    gt_train = [decodetext(char_dict, [char for char in np.trim_zeros(np.array(word), 'b')]) for word in (y_batch)]
    pre_train = [decodetext(char_dict, [char for char in np.trim_zeros(word, 'b')]) for word in decoded_]

    total_case_train += len(gt_train)
    tp_case_train = 0
    for i in range(len(pre_train)):
        if (pre_train[i].lower() == gt_train[i].lower()):
            tp_case_train += 1

    if iters % 100 == 0:
        decoded, log_prob = tf.nn.ctc_greedy_decoder(outputs,
                                                     sequence_length=[params['SEQ_LENGTH']] * len(y_batch),
                                                     merge_repeated=True)
        decoded = tf.sparse.to_dense(decoded[0]).numpy()
        print(iters, loss.numpy().round(1), [decodetext(char_dict, [char for char in np.trim_zeros(word, 'b')]) for word in decoded[:4]])

        with open('logs/loss_train.txt', 'w') as f:
            [f.write(str(s) + '\n') for s in loss_train]

        for x_test, y_test in datagen(batches=1, batch_size=124, epochs=1, dataset='test'):
            indices, values, dense_shape = sparse_tuple_from(y_test)
            y_test_sparse = tf.sparse.SparseTensor(
                indices=indices, values=values, dense_shape=dense_shape)

            logits, raw_pred, rnn_out = model(x_test)
            loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y_test_sparse,
                                                 logits=rnn_out,
                                                 label_length=[len(i) for i in y_test],
                                                 logit_length=[params['SEQ_LENGTH']] * len(y_test),
                                                 blank_index=62))
            test_loss = loss.numpy().round(1)
            loss_test.append(loss.numpy().round(1))

            decoded_test, _ = tf.nn.ctc_greedy_decoder(rnn_out,  # logits.numpy().transpose((1, 0, 2)),
                                                       sequence_length=[params['SEQ_LENGTH']] * len(y_test),
                                                       merge_repeated=True)
            decoded_test = tf.sparse.to_dense(decoded_test[0]).numpy()

            gt_ = [decodetext(char_dict, [char for char in np.trim_zeros(np.array(word), 'b')]) for word in (y_test)]
            pre_ = [decodetext(char_dict, [char for char in np.trim_zeros(word, 'b')]) for word in decoded_test]

            total_case += len(gt_)
            tp_case = 0
            for i in range(len(pre_)):
                if (pre_[i].lower() == gt_[i].lower()):
                    tp_case += 1

            print('tp_case: {0}'.format(tp_case))
            print('accuracy: {0}'.format(tp_case / len(gt_)))
            accuracy.append(tp_case / len(gt_))

            curr_epoch += 1

            # use tensorboard to plot graphs
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=curr_epoch)
                tf.summary.scalar('accuracy', tp_case_train /
                                  len(gt_train), step=curr_epoch)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss_test', test_loss, step=curr_epoch)
                tf.summary.scalar('accuracy', tp_case / len(gt_), step=curr_epoch)

            with open('logs/loss_test.txt', 'w') as file:
                [file.write(str(s) + '\n') for s in loss_test]

            with open('logs/accuracy.txt', 'w') as file:
                [file.write(str(s) + '\n') for s in accuracy]

            # Save model when the model gets a higher accuracy
            if tp_case / len(gt_) > curr_accuracy:
                curr_accuracy = tp_case / len(gt_)
                print(f'Save model {iters}')
                model.save_weights('../../../models/crnn/new_weights.h5')
    iters += 1
