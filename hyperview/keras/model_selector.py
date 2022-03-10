import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from backbone_models.swin_transformer import SwinTransformer

class SpatioTemporalModel(tf.keras.Model):

    def __init__(self, model_type, input_shape,label_shape,pretrained):
        super(SpatioTemporalModel, self).__init__()
        #https://www.philschmid.de/image-classification-huggingface-transformers-keras

        temporal_input = tf.keras.layers.Input(shape=input_shape)
        t_shape=temporal_input.shape
        input_list = tf.split(temporal_input, num_or_size_splits=int(t_shape[-1]/3), axis=-1)
        feature = tf.squeeze(tf.stack(input_list, axis=1),-4)


        backbone=BackboneModel(model_type,feature.shape[2:],pretrained)
        #backbone.build((feature.shape[0],*feature.shape[2:]))
        #backbone.compile(run_eagerly=True)
        #backbone.summary()



        multi_chanel_model = tf.keras.Sequential()
        multi_chanel_model.add(TimeDistributed(backbone, input_shape=feature.shape[1:]))
        multi_chanel_model.add(Flatten())
        multi_chanel_model.add(Dense(256, activation=tf.keras.layers.LeakyReLU()))
        multi_chanel_model.add(Dropout(0.5))
        multi_chanel_model.add(BatchNormalization())
        #multi_chanel_model.add(Dense(64, activation=tf.keras.layers.LeakyReLU()))
        #multi_chanel_model.add(Dropout(0.5))
        #multi_chanel_model.add(BatchNormalization())
        multi_chanel_model.add(Dense(16, activation=tf.keras.layers.LeakyReLU()))
        multi_chanel_model.add(Dropout(0.5))
        multi_chanel_model.add(BatchNormalization())

        multi_chanel_model.add(Dense(label_shape, activation=tf.keras.layers.LeakyReLU()))


        super(SpatioTemporalModel, self).__init__(inputs=temporal_input, outputs=multi_chanel_model(feature))


class BackboneModel(tf.keras.Model):
        def __init__(self, model_type, input_shape,pretrained):
            inp = tf.keras.layers.Input(shape=input_shape)
            model=None
            if model_type == 0:
                model=SwinTransformer('swin_tiny_224', num_classes=1000, include_top=False, pretrained=pretrained)

            if model_type == 1:
                weights = 'imagenet' if pretrained else None
                model=tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False,
                                                              classifier_activation=tf.keras.layers.LeakyReLU(),
                                                              weights=weights)

            if model_type == 2:
                weights = 'imagenet' if pretrained else None
                model=tf.keras.applications.MobileNetV3Large(input_shape=input_shape, include_top=False,
                                                              classifier_activation=tf.keras.layers.LeakyReLU(),
                                                              weights=weights)

            single_channel_header = tf.keras.Sequential()
            single_channel_header.add(Dense(256, activation=tf.keras.layers.LeakyReLU()))
            single_channel_header.add(Dropout(0.5))
            single_channel_header.add(BatchNormalization())
            #single_channel_header.add(Dense(64, activation=tf.keras.layers.LeakyReLU()))
            #single_channel_header.add(Dropout(0.5))
            #single_channel_header.add(BatchNormalization())
            single_channel_header.add(Dense(16, activation=tf.keras.layers.LeakyReLU()))
            single_channel_header.add(Dropout(0.5))
            single_channel_header.add(BatchNormalization())

            single_out = single_channel_header(model(inp))
            #backbone_with_head = tf.keras.Model(single_in, single_out)

            super(BackboneModel, self).__init__(inputs=inp, outputs=single_out)


        def compute_output_shape(self, input_shape):
            inp=tf.keras.layers.Input(shape=input_shape[1:])
            out=self.call(inp)
            return out.shape[:]
