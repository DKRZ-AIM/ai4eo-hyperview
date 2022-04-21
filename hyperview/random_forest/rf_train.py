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
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

import joblib
import optuna
from optuna.samplers import TPESampler

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


def load_data(directory: str, file_path, istrain, args):
    """Load each cube, reduce its dimensionality and append to array.

    Args:
        directory (str): Directory to either train or test set
    Returns:
        [type]: A list with spectral curve for each sample.
    """
    datalist = []
    masklist = []
    aug_datalist = []
    aug_masklist = []
    aug_labellist = []

    if istrain:
        labels = load_gt(file_path, args)

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

    if istrain:
        for i in range(args.augment_constant):
            for idx, file_name in enumerate(all_files):
                with np.load(file_name) as npz:
                    mask = npz['mask']
                    data = npz['data']
                    ma = np.max(data, keepdims=True)
                    sh = data.shape[1:]
                    max_edge = np.max(sh)
                    min_edge = np.min(sh)  # AUGMENT BY SHAPE
                    edge = min_edge  # np.random.randint(16, min_edge)
                    x = np.random.randint(sh[0] + 1 - edge)
                    y = np.random.randint(sh[1] + 1 - edge)
                    aug_data = data[:, x:(x + edge), y:(y + edge)] + np.random.uniform(-0.01, 0.01,
                                                                                       (150, edge, edge)) * ma
                    aug_mask = mask[:, x:(x + edge), y:(y + edge)] | np.random.randint(0, 1, (150, edge, edge))
                    aug_datalist.append(aug_data)
                    aug_masklist.append(aug_mask)
                    aug_labellist.append(labels[idx, :] + labels[idx, :] * np.random.uniform(-0.01, 0.01, 4))

    if istrain:
        return datalist, masklist, labels, aug_datalist, aug_masklist, np.array(aug_labellist)
    else:
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
    
    # only 11x11 patches
    #gt_file = gt_file[:650]

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

    def _random_pixel(data):
        '''draws (min_sample_size x min_sample_size) patches from each patch''' 
        
        min_edge = 11
        shape = (min_edge, min_edge)

        random_select = [np.random.choice(data[i].flatten(), min_edge*min_edge, replace=False).reshape(shape) for i in range(data.shape[0])]
        random_select = np.array(random_select)

        return random_select

    filtering = SpectralCurveFiltering()

    processed_data = []

    for idx, (data, mask) in enumerate(tqdm(zip(data_list, mask_list), total=len(data_list), 
                                        position=0, leave=True, desc="INFO: Preprocessing data ...")):

        data = data/2210 ## max-max=5419 mean-max=2210
        m = (1 - mask.astype(int))
        image = (data * m)
        #image = _random_pixel(image) 
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
        out = np.concatenate([arr,dXdl, d2Xdl2, s[:,0], s[:,1], s[:,2], s[:,3], s[:,4], real, imag], -1)

        processed_data.append(out)

    return np.array(processed_data)


def mixing_augmentation(X, y, fract, mix_const):

    mix_index_1 = np.random.randint(X.shape[0], size=int(np.floor(X.shape[0]*fract)))
    mix_index_2 = np.random.randint(X.shape[0], size=int(np.floor(X.shape[0]*fract)))

    ma_X = (1 - mix_const) * X[mix_index_1] + mix_const * (X[mix_index_2])
    ma_y = (1 - mix_const) * y[mix_index_1] + mix_const * (y[mix_index_2])

    return np.concatenate([X, ma_X], 0), np.concatenate([y, ma_y], 0)

def evaluation_score(args, y_v, y_hat, y_b, cons):
    score = 0
    for i in range(len(args.col_ix)):
        print(f'Soil idx {i} / {len(args.col_ix)-1}')
        mse_model = mean_squared_error(y_v[:, i]*cons[i], y_hat[:, i]*cons[i])
        mse_bl = mean_squared_error(y_v[:, i]*cons[i], y_b[:, i]*cons[i])

        score += mse_model / mse_bl

        print(f'Baseline MSE:      {mse_bl:.2f}')
        print(f'Model MSE: {mse_model:.2f} ({1e2*(mse_model - mse_bl)/mse_bl:+.2f} %)')
        print(f'Evaluation score: {score/len(args.col_ix)}')
    
    return score / 4       

def print_feature_importances(feature_names, importances):
    
    feats = {}
    for feature, importance in zip(feature_names, importances):
         feats[feature] = importance
    feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
    for feat in feats:
        print(f'{feat[0]}: {feat[1]}')



def predictions_and_submission(study, X_processed, X_test, y_train_col, cons, args):
   
    final_model = study.best_params["regressor"]
    if final_model == "RandomForest":
        # fit rf with best parameters on entire training data 
        optimised_model = RandomForestRegressor(n_estimators=study.best_params['n_estimators'], 
                                             max_depth=study.best_params['max_depth'],
                                             min_samples_leaf=study.best_params['min_samples_leaf'],
                                             n_jobs=-1, 
                                             criterion="squared_error")
    else:
        optimised_model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror',
                                                           n_estimators=study.best_params['n_estimators'],
                                                           eta=study.best_params['eta'],
                                                           gamma=study.best_params['gamma'],
                                                           alpha=study.best_params['alpha'],
                                                           max_depth=study.best_params['max_depth'],
                                                           min_child_weight=study.best_params['min_child_weight'],
                                                           verbosity=1))

    optimised_model.fit(X_processed, y_train_col)
    predictions = optimised_model.predict(X_test)
    
    #if args.col_ix == 4:
    #    # revert normalization of 1st and second idx 
    #    predictions[:,0] = np.expm1(predictions[:,0])
    #    predictions[:,1] = np.expm1(predictions[:,1])

    predictions = predictions * np.array(cons[:len(args.col_ix)])
    
    # calculate score on full training set
    baseline = BaselineRegressor()
    baseline.fit(X_processed, y_train_col)
    y_b = baseline.predict(X_processed)
    y_fulltrain_pred = optimised_model.predict(X_processed)

    score = evaluation_score(args, y_train_col, y_fulltrain_pred, y_b, cons)
    print(f'\nScore of best model ({final_model}) on training set: {score}\n')

    # print feature importances
    feature_names=['arr','dXdl', 'd2Xdl2', 's_0', 's_1', 's_2', 's_3', 's_4', 'real', 'imag']
    if final_model == 'RandomForest':
        importances = optimised_model.feature_importances_
        print_feature_importances(feature_names, importances)
    else:
        for i in range(len(optimised_model.estimators_)):
            importances = optimised_model.estimators_[i].feature_importances_
            
            print_feature_importances(feature_names, importances)
    
    # save the model
    if args.save_model:
        output_file = os.path.join(args.model_dir, f"{final_model}_SIMPLE_{date_time}_"\
                f"nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_"\
                f"minsl={study.best_params['min_samples_leaf']}.bin")
            
        with open(output_file, "wb") as f_out:
            joblib.dump(optimised_model, f_out)

    # only make submission file, if all 4 soil parameters are considered
    # only predictions from RF are saved!
    if len(args.col_ix) == 4 and  args.debug==False:
        submission = pd.DataFrame(data=predictions, columns=["P", "K", "Mg", "pH"])
        print(submission.head())
        if final_model=="RandomForest":
            submission.to_csv(os.path.join(args.submission_dir, f"submission_{final_model}_SIMPLE"\
                f"{date_time}_nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_"\
                f"minsl={study.best_params['min_samples_leaf']}.csv"), index_label="sample_index")
        else:
            submission.to_csv(os.path.join(args.submission_dir, f"submission_{final_model}_SIMPLE"\
                f"{date_time}_nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_"\
                f"eta={eta}_gamma={gamma}_alpha={alpha}"\
                f"minsl={study.best_params['min_child_weight']}.csv"), index_label="sample_index")


def predictions_and_submission_2(study, best_model, X_test, cons, args, min_score):

    predictions = []
    for rf in best_model:
        pp = rf.predict(X_test)
        predictions.append(pp)
    predictions = np.asarray(predictions)
    predictions = np.mean(predictions, axis=0)
    
    #if args.col_ix == 4:
    #    # revert normalization of 1st and second idx 
    #    predictions[:,0] = np.expm1(predictions[:,0])
    #    predictions[:,1] = np.expm1(predictions[:,1])

    predictions = predictions * np.array(cons[:len(args.col_ix)])


    final_model = best_model[0].__class__.__name__
    # print feature importances for Random Forest
    if final_model == "RandomForestRegressor":
        feats = {}
        importances = best_model[-1].feature_importances_
        feature_names = ['arr', 'dXdl', 'd2Xdl2', 'd3Xdl3', 'dXds1', 's_0', 
                         's_1', 's_2', 's_3', 's_4', 'real', 'imag']
        #feature_names = ['arr', 'dXdl', 'd2Xdl2', 'd3Xdl3', 'dXds1', 's_0', 
        #                 's_1', 's_2', 's_3', 's_4', 'real', 'imag',
        #                 'reals','imags', 'cDw2', 'cAw2', 'cos']
        for feature, importance in zip(feature_names, importances):
            feats[feature] = importance
        feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
        for feat in feats:
            print(f'{feat[0]}: {feat[1]}')

    # only make submission file, if all 4 soil parameters are considered
    if len(args.col_ix) == 4 and args.debug == False:
        submission = pd.DataFrame(data=predictions, columns=["P", "K", "Mg", "pH"])
        print(submission.head())
        if study is not None:
            if final_model=="RandomForestRegressor":
                submission.to_csv(os.path.join(args.submission_dir, f"submission_{final_model}_CV"\
                        f"{date_time}_nest={study.best_params['n_estimators']}_"\
                        f"maxd={study.best_params['max_depth']}_" \
                        f"minsl={study.best_params['min_samples_leaf']}_"\
                        #f"aug_con={study.best_params['augment_constant']}_"\
                        #f"aug_par={study.best_params['augment_partition']}
                        f".csv"),index_label="sample_index")
            else:
               print(f"CV submission for {final_model} not supported")
        else:
            submission.to_csv(os.path.join(args.submission_dir, "submission_best_{}.csv".format(min_score)),
                    index_label="sample_index")


def main(args):

    train_data = os.path.join(args.in_data, "train_data", "train_data")
    test_data = os.path.join(args.in_data, "test_data")
    train_gt=os.path.join(args.in_data, "train_data", "train_gt.csv")

    # load the data
    print("start loading data ...")
    start_time = time.time()
    X_train, M_train, y_train, X_aug_train, M_aug_train, y_aug_train = load_data(train_data, train_gt, True, args)
    print(f"loading train data took {time.time() - start_time:.2f}s")
    print(f"train data size: {len(X_train)}")
    if args.debug==False:
        print(f"patch size examples: {X_train[0].shape}, {X_train[500].shape}, {X_train[1000].shape}")
    
    start_time = time.time()
    X_test, M_test = load_data(test_data, None, False, args)
    print(f"loading test data took {time.time() - start_time:.2f}s")
    print(f"test data size: {len(X_test)}\n")
    
    print('Preprocess training data...')
    X_processed = preprocess(X_train, M_train)
    X_aug_processed = preprocess(X_aug_train, M_aug_train)

    print('preprocess test data ...')
    X_test = preprocess(X_test, M_test)
    y_aug_train_col = y_aug_train[:, args.col_ix]
    
    # selected set of labels
    y_train_col = y_train[:, args.col_ix]

    # only for all 4 indices
    #if args.col_ix == 4:
    #    # first and second variables are log-distributed
    #    y_aug_train_col[:,0] = np.log1p(y_aug_train_col[:,0])
    #    y_aug_train_col[:,1] = np.log1p(y_aug_train_col[:,1])
    #    y_train_col[:,0] = np.log1p(y_train_col[:,0])
    #    y_train_col[:,1] = np.log1p(y_train_col[:,1])


    cons = np.array([325.0, 625.0, 400.0, 7.8])
  
    global best_model
    best_model = None
    global min_score
    min_score = np.inf

    def objective(trial):
        global best_model
        global min_score

        print(f"\nTRIAL NUMBER: {trial.number}\n")
        # training
        kfold = KFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)
    
        random_forests = []
        baseline_regressors = []
        y_hat_bl = []
        y_hat_rf = []
        scores = []

        print("START TRAINING ...")
        for i, (ix_train, ix_valid) in enumerate(kfold.split(np.arange(0, len(y_train)))):
        
            print(f'fold {i}:')

            X_t = X_processed[ix_train]
            y_t = y_train_col[ix_train]
     
            augment_constant = trial.suggest_int('augment_constant', 0, args.augment_constant, log=False)
            augment_partition = trial.suggest_int('augment_partition', args.augment_partition[0], args.augment_partition[1], log=True)

            for idy in range(augment_constant):
                X_ta_1 = X_aug_processed[ix_train+(idy*len(y_train))]
                y_ta_1 = y_aug_train_col[ix_train+(idy*len(y_train))]
                X_t=np.concatenate((X_t,X_ta_1[-augment_partition:]),axis=0)
                y_t=np.concatenate((y_t,y_ta_1[-augment_partition:]),axis=0)


            # mixing augmentation
            if args.mix_aug:
                fract = trial.suggest_categorical('fract', args.fract)
                mix_const = trial.suggest_float('mix_const', args.fract)
                X_t, y_t = mixing_augmentation(X_t, y_t, fract, mix_const)

            X_v = X_processed[ix_valid]
            y_v = y_train_col[ix_valid]

            # baseline
            baseline = BaselineRegressor()
            baseline.fit(X_t, y_t)
            baseline_regressors.append(baseline)

            reg_name= trial.suggest_categorical("regressor", args.regressors)

            print(f"Training on {reg_name}")
            if reg_name == "RandomForest":
                n_estimators =  trial.suggest_int('n_estimators', args.n_estimators[0], args.n_estimators[1], log=True)
                max_depth =  trial.suggest_categorical('max_depth', args.max_depth)
                min_samples_leaf =  trial.suggest_categorical('min_samples_leaf', args.min_samples_leaf)

                # random forest
                model = RandomForestRegressor(n_estimators=n_estimators, 
                                           max_depth=max_depth, 
                                           min_samples_leaf=min_samples_leaf, 
                                           n_jobs=-1, 
                                           criterion="squared_error")
            else:
                n_estimators =  trial.suggest_int('n_estimators', args.n_estimators[0], args.n_estimators[1], log=True)
                eta = trial. suggest_float('eta', args.eta[0], args.eta[1], log=True)
                gamma = trial. suggest_float('gamma', args.gamma[0], args.gamma[1])
                alpha = trial. suggest_float('alpha', args.alpha[0], args.alpha[1])
                max_depth = trial. suggest_categorical('max_depth', args.max_depth)
                min_child_weight = trial. suggest_categorical('min_child_weight', args.min_child_weight)


                # xgboost
                model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror',
                                                           n_estimators=n_estimators,
                                                           eta=eta,
                                                           gamma=gamma,
                                                           alpha=alpha,
                                                           max_depth=max_depth,
                                                           min_child_weight=min_child_weight,
                                                           verbosity=1))
            
            model.fit(X_t, y_t)
            random_forests.append(model)
            print(f'{reg_name} score: {model.score(X_v, y_v)}')

            # predictions
            y_hat = model.predict(X_v)
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
        if mean_score < min_score:
            min_score = mean_score
            best_model = random_forests
            predictions_and_submission_2(None, best_model, X_test, cons, args, min_score)

        return mean_score
    
    study = optuna.create_study(sampler=TPESampler(), direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)

    # save study
    final_model = study.best_params["regressor"]

    if args.debug == False and final_model=="RandomForest":
        output_file = os.path.join(args.submission_dir, f"study_{final_model}_{date_time}_nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_minsl={study.best_params['min_samples_leaf']}.pkl")
    if args.debug == False and final_model=="XGB":
        output_file = os.path.join(args.submission_dir, f"study_{final_model}_{date_time}_nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_eta={eta}_gamma={gamma}_alpha={alpha}_minsl={study.best_params['min_child_weight']}.pkl")

    if args.debug == False:
        with open(output_file, "wb") as f_out:
            joblib.dump(study, f_out)

    # prepare submission
    print("MAKE PREDICTIONS AND PREPARE SUBMISSION")
    # train best model on full training set
    predictions_and_submission(study, X_processed, X_test, y_train_col, cons, args)
    # cross validation on validation set
    predictions_and_submission_2(study, best_model, X_test, cons, args,min_score)
    print("PREDICTIONS AND SUBMISSION FINISHED")


if __name__ == "__main__":

    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M")

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--in-data', type=str, 
            default='/p/project/hai_cons_ee/kuzu/ai4eo-hyperview/hyperview/keras/')
    parser.add_argument('--submission-dir', type=str, 
            default='/p/project/hai_cons_ee/frauke/ai4eo-hyperview/hyperview/random_forest/submissions')
    parser.add_argument('--model-dir', type=str, 
            default='/p/project/hai_cons_ee/frauke/ai4eo-hyperview/hyperview/random_forest/models')
    parser.add_argument('--save-pred', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--col-ix', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--folds', type=int, default=5)
    # model hyperparams
    parser.add_argument('--n-estimators', type=int, nargs='+', default=[500, 1000])
    parser.add_argument('--max-depth', type=int, nargs='+', default=[5, 10, 100, None])
    parser.add_argument('--max-depth-none', action='store_true', default=False)
    parser.add_argument('--min-samples-leaf', type=int, nargs='+', default=[1, 10, 50])
    parser.add_argument('--eta', type=float, nargs='+', default=[0.1, 0.5]) # default 0.3
    parser.add_argument('--gamma', type=float, nargs='+', default=[0, 1]) # default=0
    parser.add_argument('--alpha', type=float, nargs='+', default=[0, 1]) # default=0
    parser.add_argument('--min-child_weight', type=int, nargs='+', default=[1, 10, 50])
    parser.add_argument('--regressors', type=str, nargs='+', default=["RandomForest", "XGB"])
    parser.add_argument('--n-trials', type=int, default=100)
    # augmentation
    parser.add_argument('--mix-aug', action='store_true', default=False)
    parser.add_argument('--fract', type=float, nargs='+', default=[0.1])
    parser.add_argument('--mix-const', type=float, nargs='+', default=[0.05])
    parser.add_argument('--augment-constant', type=int, default=5)
    parser.add_argument('--augment-partition', type=int, nargs='+', default=[100, 350])


    args = parser.parse_args()

    # None is added to max-depth (annot be done directly -> type error)
    if args.max_depth_none:
        args.max_depth = args.max_depth + [None]

    print('BEGIN argparse key - value pairs')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('END argparse key - value pairs')
    print()

    cols = ["P205", "K", "Mg", "pH"]
    
    main(args)

