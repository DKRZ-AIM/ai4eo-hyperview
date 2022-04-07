#!/usr/bin/env python3

import os
from glob import glob
import numpy as np
import pandas as pd
import random
import time
from datetime import datetime
import argparse
from tqdm.auto import tqdm

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import joblib
import optuna

import sys

class BaselineRegressor():
    """
    Baseline regressor, which calculates the mean value of the target from the training
    data and returns it for each testing sample.
    """
    def __init__(self):
        self.mean = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.mean = np.mean(y_train, axis=0)
        self.classes_count = y_train.shape[1]
        return self

    def predict(self, X_test: np.ndarray):
        return np.full((len(X_test), self.classes_count), self.mean)


class SpectralCurveFiltering():
    """
    Create a histogram (a spectral curve) of a 3D cube, using the merge_function
    to aggregate all pixels within one band. The return array will have
    the shape of [CHANNELS_COUNT]
    """

    def __init__(self, merge_function = np.mean):
        self.merge_function = merge_function

    def __call__(self, sample: np.ndarray):
        return self.merge_function(sample, axis=(1, 2))


def load_data(directory: str, args):
    """Load each cube, reduce its dimensionality and append to array.

    Args:
        directory (str): Directory to either train or test set
    Returns:
        [type]: A list with spectral curve for each sample.
    """
    datalist = []
    masklist = []
    
    all_files = np.array(
        sorted(
            glob(os.path.join(directory, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", "")),
        )
    )

    # in debug mode, only consider first 100 patches
    if args.debug:
        all_files = all_files[:100]

    for file_name in all_files:
        with np.load(file_name) as npz:
            mask = npz['mask']
            data = npz['data']
            
            datalist.append(data)
            masklist.append(mask)

    return datalist, masklist

def load_gt(file_path: str, args):
    """Load labels for train set from the ground truth file.
    Args:
        file_path (str): Path to the ground truth .csv file.
    Returns:
        [type]: 2D numpy array with soil properties levels
    """
    gt_file = pd.read_csv(file_path)

    # in debug mode, only consider first 100 patches
    if args.debug:
        gt_file = gt_file[:100]

    labels = gt_file[["P", "K", "Mg", "pH"]].values/np.array([325.0, 625.0, 400.0, 7.8]) # normalize ground-truth between 0-1
    
    return labels

def preprocess(data_list, mask_list):

    def _shape_pad(data):
        max_edge = np.max(image.shape[1:])
        shape = (max_edge, max_edge)
        padded = np.pad(data,((0, 0),
                             (0, (shape[0] - data.shape[1])),
                             (0, (shape[1] - data.shape[2]))),
                             'wrap')
        return padded

    filtering = SpectralCurveFiltering()

    processed_data = []

    for idx, (data, mask) in enumerate(tqdm(zip(data_list, mask_list), total=len(data_list), 
                                        position=0, leave=True, desc="INFO: Preprocessing data ...")):

        data = data/2210 ## max-max=5419 mean-max=2210
        m = (1 - mask.astype(int))
        image = (data * m)
        image = _shape_pad(image)

        s = np.linalg.svd(image, full_matrices=False, compute_uv=False)

        data = np.ma.MaskedArray(data, mask)
        arr = filtering(data)

        # first gradient
        dXdl = np.gradient(arr, axis=0)

        # second gradient
        d2Xdl2 = np.gradient(dXdl, axis=0)

        # fourier transform
        fft = np.fft.fft(arr)
        real = np.real(fft)
        imag = np.imag(fft)

        # final input matrix
        out = np.concatenate([arr,dXdl, d2Xdl2, s[:,0], s[:,1], s[:,3], s[:,4], real, imag], -1)

        processed_data.append(out)

    return np.array(processed_data)


def mixing_augmentation(X, y, fract=0.1):

    mix_const = 0.05
    mix_index_1 = np.random.randint(X.shape[0], size=int(np.floor(X.shape[0]*fract)))
    mix_index_2 = np.random.randint(X.shape[0], size=int(np.floor(X.shape[0]*fract)))

    ma_X = (1 - mix_const) * X[mix_index_1] + mix_const * (X[mix_index_2])
    ma_y = (1 - mix_const) * y[mix_index_1] + mix_const * (y[mix_index_2])

    return np.concatenate([X, ma_X], 0), np.concatenate([y, ma_y], 0)

def evaluation_score(args, y_v, y_hat, y_b, cons):
    score = 0
    for i in range(len(args.col_ix)):
        print(f'Soil idx {i} / {len(args.col_ix)-1}')
        mse_rf = mean_squared_error(y_v[:, i]*cons[i], y_hat[:, i]*cons[i])
        mse_bl = mean_squared_error(y_v[:, i]*cons[i], y_b[:, i]*cons[i])

        score += mse_rf / mse_bl

        print(f'Baseline MSE:      {mse_bl:.2f}')
        print(f'Random Forest MSE: {mse_rf:.2f} ({1e2*(mse_rf - mse_bl)/mse_bl:+.2f} %)')
        print(f'Evaluation score: {score/len(args.col_ix)}')
    
    return score / 4       

def predictions_and_submission(study, X_processed, X_test, y_train_col, cons, args):
    #predictions = []
    #for rf in random_forests:
    #    pp = rf.predict(X_test)
    #    predictions.append(pp)

    #predictions = np.asarray(predictions)
    #predictions = np.mean(predictions, axis=0)
    #predictions = predictions * np.array(cons[:len(args.col_ix)])
    
    # fit rf with best parameters on entire training data 
    optimised_rf = RandomForestRegressor(n_estimators=study.best_params['n_estimators'], n_jobs=-1, criterion="squared_error")
    optimised_rf.fit(X_processed, y_train_col)
    predictions = optimised_rf.predict(X_test)
    
    # save the model
    if args.save_model:
        output_file = os.path.join(args.model_dir, f"rf_{date_time}_n_est={args.n_estimators}.bin")
            
        with open(output_file, "wb") as f_out:
            joblib.dump(optimised_rf, f_out)

    # only make submission file, if all 4 soil parameters are considered
    if len(args.col_ix) == 4:
        submission = pd.DataFrame(data=predictions, columns=["P", "K", "Mg", "pH"])
        print(submission.head())
        submission.to_csv(os.path.join(args.submission_dir, f"submission_rf_{date_time}_n_est={study.best_params['n_estimators']}.csv"), index_label="sample_index")
        return predictions, submission
    
    return predictions

def main(args):
    

    train_data = os.path.join(args.in_data, "train_data", "train_data")
    test_data = os.path.join(args.in_data, "test_data")
    
    # load the data
    print("start loading data ...")
    start_time = time.time()
    X_train, M_train = load_data(train_data, args)
    print(f"loading train data took {time.time() - start_time:.2f}s")
    print(f"train data size: {len(X_train)}")
    if args.debug==False:
        print(f"patch size examples: {X_train[0].shape}, {X_train[500].shape}, {X_train[1000].shape}")
    
    start_time = time.time()
    y_train = load_gt(os.path.join(args.in_data, "train_data", "train_gt.csv"), args)
    X_test, M_test = load_data(test_data, args)
    print(f"loading test data took {time.time() - start_time:.2f}s")
    print(f"test data size: {len(X_test)}\n")
    
    print('Preprocess training data...')
    X_processed = preprocess(X_train, M_train)
    # selected set of labels
    y_train_col = y_train[:, args.col_ix]

    cons = np.array([325.0, 625.0, 400.0, 7.8])
   
    def objective(trial):
        
        print(f"\nTRIAL NUMBER: {trial.number}\n")
        # training
        kfold = KFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)
    
        random_forests = []
        baseline_regressors = []
        y_hat_bl = []
        y_hat_rf = []
        scores = []
        final_score = np.inf

        print("START TRAINING ...")
        for i, (ix_train, ix_valid) in enumerate(kfold.split(np.arange(0, len(y_train)))):
        
            print(f'fold {i}:')

            X_t = X_processed[ix_train]
            y_t = y_train_col[ix_train]
        
            # mixing augmentation
            if args.mix_aug:
                X_t, y_t = mixing_augmentation(X_t, y_t)

            X_v = X_processed[ix_valid]
            y_v = y_train_col[ix_valid]

            # baseline
            baseline = BaselineRegressor()
            baseline.fit(X_t, y_t)
            baseline_regressors.append(baseline)

            n_estimators =  trial.suggest_int('n_estimators', args.n_estimators[0], args.n_estimators[1])

            # random forest
            rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, criterion="squared_error")
            rf.fit(X_t, y_t)
            random_forests.append(rf)
            print(f'Random Forest score: {rf.score(X_v, y_v)}')

            # predictions
            y_hat = rf.predict(X_v)
            y_b = baseline.predict(X_v)

            y_hat_bl.append(y_b)
            y_hat_rf.append(y_hat)

            # evaluation score
            score = evaluation_score(args, y_v, y_hat, y_b, cons)
            scores.append(score)
            print(scores)
    
        print("END TRAINING")
        # final score
        mean_score = np.mean(np.array(scores))
        print(f'mean score: {mean_score}\n')
        
        # best models
        if mean_score < final_score:
            final_score = mean_score
            trial = trial.number

        print(f'final score {final_score} from trial {trial}\n')
        
        return mean_score
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)
    
    # save study
    output_file = os.path.join(args.submission_dir, f"study_rf_{date_time}_n_est={study.best_params['n_estimators']}.pkl")
    with open(output_file, "wb") as f_out:
        joblib.dump(study, f_out)

    # prepare submission
    print("MAKE PREDICTIONS AND PREPARE SUBMISSION")
    print('preprocess test data ...')
    X_test = preprocess(X_test, M_test)
    
    if args.col_ix == 4:
        predictions, submission = predictions_and_submission(study, X_processed, X_test, y_train_col, cons, args)
    else:
        predictions = predictions_and_submission(study, X_processed, X_test, y_train_col, cons, args)
    print("PREDICTIONS AND SUBMISSION FINISHED")

    # save predictions
    if args.save_pred:
        pass


if __name__ == "__main__":

    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--in-data', type=str, default='/p/project/hai_cons_ee/kuzu/ai4eo-hyperview/hyperview/keras/')
    parser.add_argument('--submission-dir', type=str, default='/p/project/hai_cons_ee/frauke/ai4eo-hyperview/hyperview/random_forest/submissions')
    parser.add_argument('--model-dir', type=str, default='/p/project/hai_cons_ee/frauke/ai4eo-hyperview/hyperview/random_forest/models')
    parser.add_argument('--save-pred', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--col-ix', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--mix-aug', action='store_true', default=False)
    # model hyperparams
    parser.add_argument('--n-estimators', type=int, nargs='+', default=[500, 1000])
    parser.add_argument('--n-trials', type=int, default=200)

    args = parser.parse_args()

    output = os.path.join(args.submission_dir, f"out_{date_time}_n_est={args.n_estimators}")
    sys.stdout = open(output, 'w')
    
    print('BEGIN argparse key - value pairs')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('END argparse key - value pairs')
    print()

    cols = ["P205", "K", "Mg", "pH"]

    main(args)

    sys.stdout.close()
