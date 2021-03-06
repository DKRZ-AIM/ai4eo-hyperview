from tensorflow.keras.utils import Sequence
import tensorflow.python.ops.numpy_ops.np_config as np_config
import tensorflow as tf
import pandas as pd
from functools import partial
import albumentations as A
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class DataGenerator():
    def __init__(self,train_dir, label_dir, eval_dir,valid_size=0.2,channel_type=0, batch_size=16):
        """Constructor.
                """
        self.train_dir=train_dir
        self.label_dir=label_dir
        self.eval_dir=eval_dir
        self.batch_size = batch_size

        train_stats_log = '{}/stats.npy'.format(train_dir)
        eval_stats_log = '{}/stats.npy'.format(eval_dir)

        if (not os.path.exists(train_stats_log)):
            train_stats = DataGenerator._get_stats(train_dir)
            np.save(train_stats_log, train_stats)
        if (not os.path.exists(eval_stats_log)):
            eval_stats = DataGenerator._get_stats(eval_dir)
            np.save(eval_stats_log, eval_stats)

        self.train_stats = np.load(train_stats_log)
        self.eval_stats = np.load(eval_stats_log)

        train_files = DataGenerator._load_data(train_dir)
        train_labels = DataGenerator._load_gt(label_dir)
        train_files, valid_files, train_labels, valid_labels = train_test_split(train_files, train_labels, test_size = valid_size, random_state = 42)
        test_files, valid_files, test_labels, valid_labels = train_test_split(valid_files, valid_labels,test_size=0.5, random_state=42)

        eval_files = DataGenerator._load_data(eval_dir)
        eval_labels=np.zeros(eval_files.shape)

        self.train_reader = DataGenerator._get_data_reader(train_files,train_labels,batch_size,True,channel_type=channel_type,stats=self.train_stats)
        self.valid_reader = DataGenerator._get_data_reader(valid_files, valid_labels,batch_size, True,channel_type=channel_type,stats=self.train_stats)
        self.test_reader = DataGenerator._get_data_reader(test_files, test_labels, batch_size, False,channel_type=channel_type,stats=self.eval_stats)
        self.eval_reader = DataGenerator._get_data_reader(eval_files, eval_labels,batch_size, False,channel_type=channel_type,eval=True,stats=self.eval_stats)

        self.image_shape, self.label_shape = DataGenerator._get_dataset_features(self.valid_reader)


    @staticmethod
    def _get_dataset_features(reader):
        for feature, mask, in reader.take(1):
            image_shape=tuple([*feature.shape[1:]])
            label_shape=mask.shape[-1]
            return image_shape,label_shape,

    @staticmethod
    def _get_data_reader(files, labels, batch_size, transform, eval=False,stats=None,channel_type=0):

        dataset = tf.data.Dataset.from_tensor_slices((files,labels))
        dataset = dataset.interleave(lambda x,y: DataGenerator._deparse_single_image(x, y,stats,channel_type),cycle_length=batch_size,num_parallel_calls=tf.data.AUTOTUNE)
        if not eval:
            dataset = dataset.shuffle(buffer_size=len(files), reshuffle_each_iteration=True)
        dataset = dataset.map(partial(DataGenerator._trans_single_image, transform=transform,eval=eval),num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=False,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        #if batch_size<2:
        return dataset
        #else:
        #    options = tf.data.Options()
        #    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        #    return dataset.with_options(options)

    @staticmethod
    def _load_gt(file_path: str):
        """Load labels for train set from the ground truth file.
        Args:
            file_path (str): Path to the ground truth .csv file.
        Returns:
            [type]: 2D numpy array with soil properties levels
        """
        gt_file = pd.read_csv(file_path)
        labels = gt_file[["P", "K", "Mg", "pH"]].values #/ np.array([325,625,400,7.8])
        return labels

    @staticmethod
    def _load_data(directory: str):
        all_files = np.array(
            sorted(
                glob(os.path.join(directory, "*.npz")),
                key=lambda x: int(os.path.basename(x).replace(".npz", "")),
            )
        )
        return all_files

    @staticmethod
    def _shape_pad(data,shape):
        return np.pad(data,
                      ((0, 0),
                       (0, (shape[0] - data.shape[1])),
                       (0, (shape[1] - data.shape[2]))),
                      'constant', constant_values=0)

    @staticmethod
    def _deparse_single_image(filename,label,stats=None,channel_type=0):
        filtering = SpectralCurveFiltering()
        def _read_npz(filename):
            with np.load(filename.numpy()) as npz:

                print(np.max(stats[2])*np.max(stats[0]))
                PAD_CONST_1 = 6715800
                PAD_CONST_2 = int(1228800)
                data=npz['data']/np.max(stats[-1])
                mask = npz['mask']
                data = np.ma.MaskedArray(data,mask)


                if channel_type==1:
                    arr = filtering(data)
                    arr= arr/ np.linalg.norm(arr)
                    #arr = np.ma.mean(data,axis=(1,2))
                    var = np.ma.var(data,(1,2))
                    var = var / np.linalg.norm(var)

                    dXdl = np.gradient(arr, axis=0)
                    #dXdl = dXdl / np.linalg.norm(dXdl)

                    d2Xdl2 = np.gradient(dXdl, axis=0)
                    #d2Xdl2 = d2Xdl2 / np.linalg.norm(d2Xdl2)

                    #q1 = np.percentile(data, 25, axis=(1, 2))
                    #q1 = q1 / np.linalg.norm(q1)

                    #q2 = np.percentile(data, 50, axis=(1, 2))
                    #q2 = q2 / np.linalg.norm(q2)

                    #q3 = np.percentile(data, 75, axis=(1, 2))
                    #q3 = q3 / np.linalg.norm(q3)

                    fft = np.fft.fft(arr)
                    real=np.real(fft)
                    #real = real / np.linalg.norm(real)
                    imag=np.imag(fft)
                    #imag = imag / np.linalg.norm(imag)

                    arr=np.expand_dims(arr,1)

                    var = np.expand_dims(var,1)
                    #q1 = np.expand_dims(q1,1)
                    #q2 = np.expand_dims(q2,1)
                    #q3 = np.expand_dims(q3,1)
                    dXdl = np.expand_dims(dXdl,1)
                    d2Xdl2 = np.expand_dims(d2Xdl2,1)
                    real = np.expand_dims(real,1)
                    imag = np.expand_dims(imag,1)
                    out=np.concatenate([arr,var,dXdl,d2Xdl2,real,imag],1)

                    return out
                elif channel_type==3:
                    data = data.flatten('F')
                    data = data[~data.mask]
                    idx = np.random.randint(int(len(data)/150), size=128)
                    data_list=[]
                    for id in idx:
                        da=data[id*150:id*150+150]
                        dXdl = np.gradient(da, axis=0)
                        dXdl2 = np.gradient(dXdl, axis=0)
                        fft = np.fft.fft(da)
                        real = np.real(fft)
                        imag = np.imag(fft)

                        da=np.expand_dims(da,1)
                        dXdl = np.expand_dims(dXdl, 1)
                        dXdl2 = np.expand_dims(dXdl2, 1)
                        real = np.expand_dims(real, 1)
                        imag = np.expand_dims(imag, 1)

                        data_list.append(da)
                        data_list.append(dXdl)
                        data_list.append(dXdl2)
                        data_list.append(real)
                        data_list.append(imag)
                        #TODO this can be reimplemented for 2D image reading

                    out=np.concatenate(data_list,1)

                    return out

                elif channel_type==2:
                    #data = data / stats[-1]
                    data = data.flatten('F')
                    data = data[~data.mask]
                    if len(data)<PAD_CONST_2:
                        shift = np.random.randint(10) * 150
                        data = np.pad(data, ((shift, PAD_CONST_2 - len(data)-shift)), 'wrap')
                    else:
                        le=np.floor(len(data)/150)
                        shift = np.random.randint(le) * 150
                        if shift + PAD_CONST_2<len(data):
                            data=data[shift:shift+PAD_CONST_2]
                        else:
                            data = data[:PAD_CONST_2]
                #    #pde = pde.T
                    return data
        [image ] = tf.py_function(_read_npz, [filename], [tf.float32])

        return tf.data.Dataset.from_tensors((image, label,filename))

    @staticmethod
    def _trans_single_image(feature,label,filename,transform=None,eval=True):
        def _aug_fn(feature):

            if not eval:
                sample = np.random.uniform(low=-0.0001, high=0.0001, size=feature.shape)
                feature=feature + sample

            #feature=feature.transpose((2, 0, 1))
            #feature = np.nan_to_num(feature, nan=np.finfo(float).eps, posinf=np.finfo(float).eps, neginf=-np.finfo(float).eps)
            feature = tf.cast(feature, tf.float32)
            return feature
        if transform is not None:
            feature = tf.numpy_function(func=_aug_fn, inp=[feature], Tout=[tf.float32])
        if not eval:
            return tf.cast(feature, tf.float32), tf.cast(label, tf.float32)
        else:
            return tf.cast(feature, tf.float32), tf.cast(label, tf.float32),filename

    @staticmethod
    def _get_stats(directory: str):
        all_files = np.array(
            sorted(
                glob(os.path.join(directory, "*.npz")),
                key=lambda x: int(os.path.basename(x).replace(".npz", "")),
            )
        )
        data=[]
        for file_name in all_files:
            with np.load(file_name) as npz:
                arr = np.ma.MaskedArray(**npz)
                d = np.max(arr, (1, 2))
                data.append(d)
        data=np.array(data)
        max_data=np.max(data,0)

        fst_moment = 0
        snd_moment = 0
        total_pixel_sum = 0.0
        total_pixel_count = 0.0
        total_pixel_sum_sq = 0.0


        for file_name in all_files:
            with np.load(file_name) as npz:
                br=np.tile(max_data[:, np.newaxis, np.newaxis], npz['data'].shape[1:])
                arr = np.ma.MaskedArray(**npz)
                arr=arr/br
                #d = np.max(arr, (1, 2))
                #data.append(d)
                nb_pixels=np.sum(1 - npz['mask'][0].astype(int)).astype(np.float)
                total_pixel_sum = np.sum(arr.astype(np.float), (1, 2))
                total_pixel_sum_sq = np.sum(arr.astype(np.float) ** 2, (1, 2))
                fst_moment = (total_pixel_count * fst_moment + total_pixel_sum) / (total_pixel_count + nb_pixels)
                snd_moment = (total_pixel_count * snd_moment + total_pixel_sum_sq) / (total_pixel_count + nb_pixels)
                total_pixel_count += nb_pixels

                #https://www.binarystudy.com/2021/04/how-to-calculate-mean-standard-deviation-images-pytorch.html
        mean, std = fst_moment, np.sqrt(snd_moment - fst_moment ** 2)

        #total_mean = total_pixel_sum / total_pixel_count
        #total_var = (total_pixel_sum_sq / total_pixel_count) - (total_mean ** 2)
        #total_std = np.sqrt(total_var)
        return np.array([mean, std, max_data])

    @staticmethod
    def _init_transform(image_shape, train_stats, eval_stats):
        train_transform = A.Compose([
            A.Resize(image_shape[0], image_shape[1]),
            #A.Normalize(mean=train_stats[0], std=train_stats[1], max_pixel_value=np.max(train_stats[2])),
            A.GaussNoise(var_limit=0.0025),
            A.RandomRotate90(),
            A.Rotate(),
            A.RandomResizedCrop(image_shape[0], image_shape[1], ratio=(0.98, 1.02), p=0.5),
            A.Flip(),
            A.ShiftScaleRotate(rotate_limit=0, shift_limit_x=0.02, shift_limit_y=0.02),
            # A.RandomBrightnessContrast(),

        ])

        valid_transform = A.Compose([
            A.Resize(image_shape[0], image_shape[1]),
            #A.Normalize(mean=train_stats[0], std=eval_stats[0], max_pixel_value=np.max(train_stats[2])),
            A.RandomRotate90(),
            A.Flip(),
        ])

        eval_transform = A.Compose([
            A.Resize(image_shape[0], image_shape[1]),
            #A.Normalize(mean=eval_stats[0], std=eval_stats[1], max_pixel_value=np.max(eval_stats[2]))
            ])

        return train_transform, valid_transform, eval_transform

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




