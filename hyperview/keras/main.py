from data_loader import DataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Nadam, Adam
import argparse
import os
from math import floor,ceil
import numpy as np
from tqdm.auto import tqdm
import csv
from model_selector import SpatioMultiChannellModel
import matplotlib.pyplot as plt
import keras.backend as K
import pandas as pd
import tensorflow_addons as tfa
np.random.seed(1)
tf.random.set_seed(2)


parser = argparse.ArgumentParser(description='HyperView')

parser.add_argument('-m', '--model-type', default=1, type=int, metavar='MT', help='0: X,  1: Y, 2: Z,')
parser.add_argument('-c', '--channel-type', default=1, type=int, metavar='CT', help='0: X,  1: Y, 2: Z,')
parser.add_argument('--start-epoch', default=0, type=int, metavar='SE', help='start epoch (default: 0)')
parser.add_argument('--num-epochs', default=1, type=int, metavar='NE', help='number of epochs to train (default: 120)')
parser.add_argument('--num-workers', default=4, type=int, metavar='NW', help='number of workers in training (default: 8)')
parser.add_argument('-b','--batch-size', default=16, type=int, metavar='BS', help='number of batch size (default: 32)')
parser.add_argument('-w','--width', default=64, type=int, metavar='BS', help='number of widthxheight size (default: 32)')
parser.add_argument('-l','--learning-rate', default=0.01, type=float, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--weights-dir', default='None', type=str, help='Weight Directory (default: modeldir)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate the model (it requires the wights path to be given')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true', help='pretrained or not')
parser.add_argument('--cuda', default='all', type=str, help=' cuda devices (default: 0)')

parser.add_argument('--train-dir', default='train_data/train_data/', type=str, help='path to the data directory')
parser.add_argument('--label-dir', default='train_data/train_gt.csv', type=str, help='path to the data directory')
parser.add_argument('--eval-dir', default='test_data/', type=str, help='path to the data directory')


parser.add_argument('--out-dir', default='modeldir/', type=str, help='Out Directory (default: modeldir)')
parser.add_argument('--log-file', default='performance-logs.csv', type=str, help='path to log dir (default: logdir)')

args = parser.parse_args()


def main():
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    image_shape = (args.width, args.width)
    dataset = DataGenerator(args.train_dir, args.label_dir, args.eval_dir,
                            valid_size=0.24,
                            image_shape=image_shape,
                            batch_size=args.batch_size)



    experiment_log = '{}/m_{}_c_{}_b_{}_lr_{}_p_{}_w_{}'.format(args.out_dir, args.model_type,args.channel_type, args.batch_size, args.learning_rate, args.pretrained,args.width)
    model = SpatioMultiChannellModel(args.model_type,args.channel_type, dataset.image_shape, dataset.label_shape, pretrained=args.pretrained)
    model=train_model(model, dataset, experiment_log, warmup=True)
    model=train_model(model, dataset, experiment_log, warmup=False)
    model.load_weights('{}_model_best.h5'.format(experiment_log))
    evaluate_model(model, dataset)
    create_submission(model, dataset.eval_reader,experiment_log)


def train_model(model, dataset, log_args, warmup=True):
    #strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    #with strategy.scope():
        if warmup:
            print('\n\nWARM-UP SESSION STARTED!\n\n')
            for idx in range(len(model.submodules)):
                if 'backbone_model' in model.submodules[idx].name:
                    model.submodules[idx].trainable=False
                    for idy in range(len(model.submodules[idx].layers)): model.submodules[idx].layers[idy].trainable = False

            learning_rate = args.learning_rate / 10
            num_epochs = ceil(args.num_epochs / 15)

        else:
            print('\n\nTRAINING SESSION STARTED!\n\n')
            for idx in range(len(model.submodules)):
                if 'backbone_model' in model.submodules[idx].name:
                    model.submodules[idx].trainable = True
                    for idy in range(len(model.submodules[idx].layers)): model.submodules[idx].layers[idy].trainable = True
            model.trainable=True
            learning_rate = args.learning_rate
            num_epochs = args.num_epochs


        #maximal_learning_rate=learning_rate*100
        #clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=learning_rate,
        #                                  maximal_learning_rate= maximal_learning_rate,
        #                                  scale_fn=lambda x: 1 / (2. ** (x - 1)),
        #                                  step_size=250
        #                                  )

        optimizer = Adam(learning_rate=learning_rate)
        #moving_avg_optimizer = tfa.optimizers.SWA(optimizer)


        mse_total = custom_mse()
        mse0 = custom_mse(idx=0)
        mse1 = custom_mse(idx=1)
        mse2 = custom_mse(idx=2)
        mse3 = custom_mse(idx=3)
        # mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        # lossWeights = {"total": 1, "P": 0 / 1100, "K": 0 / 2500, "Mg": 0 / 2000, "pH": 0 / 3}

        losses = {"total": mse_total, "P": mse0,"K": mse1,"Mg": mse2,"pH": mse3}
        lossWeights = {"total": 0, "P": 0.25 , "K": 0.25 , "Mg": 0.25 , "pH": 0.25 }
        model.compile(optimizer=optimizer, loss=losses,loss_weights=lossWeights, run_eagerly=True)

        callbacks = [
                ReduceLROnPlateau(verbose=1),
                EarlyStopping(patience=25),
                ModelCheckpoint(#update_weights=True,
                    filepath='{}_model_best.h5'.format(log_args),
                    monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True),
                ]

        history = model.fit(dataset.train_reader,
                                epochs=num_epochs,
                                workers=args.num_workers,
                                callbacks=callbacks,
                                use_multiprocessing=True,
                                shuffle=True,
                                validation_data=dataset.valid_reader)

        loss_log = '{}_total_loss.jpg'.format(log_args)
        print_history(history, 'loss', loss_log)
        loss_log = '{}_P_loss.jpg'.format(log_args)
        print_history(history, 'P_loss', loss_log)
        loss_log = '{}_K_loss.jpg'.format(log_args)
        print_history(history, 'K_loss', loss_log)
        loss_log = '{}_Mg_loss.jpg'.format(log_args)
        print_history(history, 'Mg_loss', loss_log)
        loss_log = '{}_pH_loss.jpg'.format(log_args)
        print_history(history, 'pH_loss', loss_log)

        return model

def evaluate_model(model, generators, logging=True):

    print('\n\nEVALUATION SESSION STARTED!\n\n')
    tr_loss = challenge_eval(model,generators.train_reader)
    val_loss = challenge_eval(model,generators.valid_reader)
    te_loss = challenge_eval(model, generators.test_reader)

    print('TOTAL LOSS:  Training: {}, Validation: {}, Test: {}'.format(tr_loss[0],val_loss[0],te_loss[0]))
    #tr_loss = model.evaluate(generators.train_reader)
    #val_loss = model.evaluate(generators.valid_reader)
    if logging:
        header = ['out_dir','m','c','b','l','p','wxh', 'train_loss', 'valid_loss', 'P','P_val','K','K_val', 'Mg','Mg_val','pH', 'pH_val','test_loss','P_test','K_test','Mg_test','pH_test']
        info = [args.out_dir, args.model_type,args.channel_type,args.batch_size,args.learning_rate,args.pretrained,args.width,
                tr_loss[0], val_loss[0], tr_loss[1], val_loss[1],tr_loss[2], val_loss[2], tr_loss[3], val_loss[3],tr_loss[4], val_loss[4],te_loss[0],te_loss[1],te_loss[2],te_loss[3],te_loss[4]]
        if not os.path.exists(args.out_dir+'/'+args.log_file):
            with open(args.out_dir+'/'+args.log_file, 'w') as file:
                logger = csv.writer(file)
                logger.writerow(header)
                logger.writerow(info)
        else:
            with open(args.out_dir+'/'+args.log_file, 'a') as file:
                logger = csv.writer(file)
                logger.writerow(info)

def create_submission(model, reader,log_args):
    print('\n\nSUBMISSION SESSION STARTED!\n\n')
    predictions = []
    files = []
    for X, Y, file_name  in reader:
        y_pred = model.predict(X)
        y_pred = y_pred[0]
        #y_pred = np.concatenate(y_pred,axis=1)
        if len(predictions)==0:
            predictions=y_pred
            files=file_name.numpy()
        else:
            predictions=np.concatenate((predictions,y_pred),axis=0)
            files=np.concatenate((files,file_name.numpy()))

    sample_index = np.expand_dims(np.array([int(os.path.basename(f.decode('utf-8')).replace(".npz", "")) for f in files]),1)
    predictions = np.concatenate((sample_index, predictions), axis=1)

    submission = pd.DataFrame(data=predictions, columns=['temp_index',"P", "K", "Mg", "pH"])
    submission=submission.sort_values(by='temp_index',ascending=True)
    submission=submission.drop(columns='temp_index')
    submission.to_csv('{}_submission.csv'.format(log_args), index_label="sample_index")

def challenge_eval(model, reader):
    predictions = []
    ground_truth = []
    for X, Y  in reader:
        y_pred = model.predict(X)
        y_pred = y_pred[0]
        #y_pred = np.concatenate(y_pred,axis=1)
        if len(predictions)==0:
            predictions = y_pred
            ground_truth=Y.numpy()
        else:
            ground_truth =np.concatenate((ground_truth,Y.numpy()),axis=0)
            predictions=np.concatenate((predictions,y_pred),axis=0)


    mse = np.mean((ground_truth - predictions) ** 2, axis=0)
    scores = mse / np.array([1100.0, 2500.0, 2000.0, 3.0])
    # Calculate the final score
    final_score = np.mean(scores)

    return np.concatenate((np.array([final_score]),scores),axis=0)



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


def custom_mse(div_factor=np.array([1100.0, 2500.0, 2000.0, 3.0]), idx=None):
    @tf.function
    def mse_1(y_true,y_pred):
        divider = tf.constant(div_factor, dtype=tf.float32)
        if idx is not None:
            y_true=y_true[:,idx]
            divider = divider[idx]

        loss = K.square(y_pred - y_true)
        loss = tf.math.divide(loss , divider)
        loss = K.mean(loss)
        return loss
    return mse_1


if __name__ == '__main__':
    main()