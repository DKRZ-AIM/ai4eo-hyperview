from data_loader import DataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Nadam, Adam
import argparse
import os
from math import floor
import numpy as np
from model_selector import SpatioTemporalModel
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='HyperView')

parser.add_argument('-m', '--model-type', default=2, type=int, metavar='MT', help='0: X,  1: Y, 2: Z,')
parser.add_argument('--start-epoch', default=0, type=int, metavar='SE', help='start epoch (default: 0)')
parser.add_argument('--num-epochs', default=15, type=int, metavar='NE', help='number of epochs to train (default: 120)')
parser.add_argument('--num-workers', default=4, type=int, metavar='NW', help='number of workers in training (default: 8)')
parser.add_argument('-b','--batch-size', default=4, type=int, metavar='BS', help='number of batch size (default: 32)')
parser.add_argument('-l','--learning-rate', default=0.2, type=float, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--weights-dir', default='None', type=str, help='Weight Directory (default: modeldir)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate the model (it requires the wights path to be given')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true', help='pretrained or not')
parser.add_argument('--cuda', default='all', type=str, help=' cuda devices (default: 0)')

parser.add_argument('--train-dir', default='/local_home/kuzu_ri/GIT_REPO/ai4eo-hyperview/train_data/train_data/', type=str, help='path to the data directory')
parser.add_argument('--label-dir', default='/local_home/kuzu_ri/GIT_REPO/ai4eo-hyperview/train_data/train_gt.csv', type=str, help='path to the data directory')
parser.add_argument('--eval-dir', default='/local_home/kuzu_ri/GIT_REPO/ai4eo-hyperview/test_data/', type=str, help='path to the data directory')


parser.add_argument('--out-dir', default='modeldir/', type=str, help='Out Directory (default: modeldir)')
parser.add_argument('--log-file', default='performance-logs.csv', type=str, help='path to log dir (default: logdir)')



args = parser.parse_args()
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
print('\n\n\n NUMBER OF DEVICES: {}\n\n\n'.format(strategy.num_replicas_in_sync))


def main():

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        experiment_log = '{}/m_{}_b_{}_lr_{}'.format(args.out_dir, args.model_type,args.batch_size,args.learning_rate)


        dataset=DataGenerator(args.train_dir,args.label_dir,args.eval_dir,
                              valid_size=0.2,
                              image_shape=(128,128),
                              batch_size=args.batch_size)

        tf.config.run_functions_eagerly(True)
        model = SpatioTemporalModel(args.model_type,dataset.image_shape,dataset.label_shape,pretrained=args.pretrained)
        model=train_model(model, dataset, experiment_log, warmup=True)
        train_model(model, dataset, experiment_log, warmup=False)


def train_model(model, dataset, log_args, warmup=True):
    if warmup:
        print('WARM-UP SESSION STARTED!')
        learning_rate = args.learning_rate / 10
        num_epochs = floor(args.num_epochs / 15)
    else:
        learning_rate = args.learning_rate
        num_epochs = args.num_epochs
        print('TRAINING SESSION STARTED!')

    optimizer = Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    def custom_mse(y_true, y_pred):
        loss = loss_object(y_true, y_pred) / tf.constant([1100.0, 2500.0, 2000.0, 3.0])
        loss = tf.reduce_mean(loss)
        return loss

    model.compile(optimizer=optimizer, loss=custom_mse, metrics=['mae', 'mse'], run_eagerly=True)

    callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=25),
            ModelCheckpoint(
                '{}_model_best.h5'.format(log_args),
                monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False),
            ]

    history = model.fit(dataset.train_reader,
                            epochs=num_epochs,
                            workers=args.num_workers,
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            shuffle=True,
                            validation_data=dataset.valid_reader)

    loss_log = '{}_model_loss.jpg'.format(log_args)
    print_history(history, 'loss', loss_log)
    return model

def print_history(history, type, file_name):
    fig = plt.figure()
    plt.plot(history.history['{}'.format(type)])
    plt.plot(history.history['val_{}'.format(type)])
    plt.title('model {}'.format(type))
    plt.ylabel('{}'.format(type))
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.grid(True)
    fig.savefig(file_name, dpi=fig.dpi)

if __name__ == '__main__':
    main()
