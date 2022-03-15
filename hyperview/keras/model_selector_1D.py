import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from backbone_models.swin_transformer import SwinTransformer
from backbone_models.mobile_vit import MobileVit
from backbone_models.vit import ViT
from backbone_models.dense_net import DenseNet
from tensorflow.keras import activations
import os


class SpatioMultiChannellModel(tf.keras.Model):

    def __init__(self, model_type, channel_type,input_shape,label_shape,pretrained):
        super(SpatioMultiChannellModel, self).__init__()
        #https://keras.io/examples/vision/vivit/
        #https://www.philschmid.de/image-classification-huggingface-transformers-keras

        input = tf.keras.layers.Input(shape=input_shape)
        if model_type==1:
            if channel_type==1:
                fet_out=SpatioMultiChannellModel.model_builder_1(label_shape, input)
            elif channel_type == 2:
                fet_out = SpatioMultiChannellModel.model_builder_4(label_shape, input)
        elif model_type == 2:
            if channel_type==1:
                fet_out=SpatioMultiChannellModel.model_builder_2(label_shape, input)
            elif channel_type == 2:
                fet_out=SpatioMultiChannellModel.model_builder_3(label_shape, input)



        [P_logit,K_logit,Mg_logit,pH_logit]=tf.unstack(fet_out, axis=-1)
        #fet_out = Layer(trainable=False,name='total')(fet_out)
        P_out = Activation(activation=activations.linear,name='P')(P_logit)
        K_out = Activation(activation=activations.linear, name='K')(K_logit)
        Mg_out = Activation(activation=activations.linear, name='Mg')(Mg_logit)
        pH_out = Activation(activation=activations.linear, name='pH')(pH_logit)


        super(SpatioMultiChannellModel, self).__init__(inputs=input, outputs=[fet_out, P_out, K_out, Mg_out, pH_out])

    @staticmethod
    def model_builder_1(label_shape, temporal_input):
        #feature = tf.squeeze(tf.stack(input_list, axis=1), -4)
        temporal_input = tf.transpose(temporal_input, (0, 2, 1))
        multi_chanel_model = tf.keras.Sequential(name='total')
        multi_chanel_model.add(tf.keras.layers.InputLayer(temporal_input.shape[1:]))
        multi_chanel_model.add(tf.keras.layers.Conv1D(1, 3,padding='valid',activation='swish',input_shape=temporal_input.shape[1:]))
        multi_chanel_model.add(tf.keras.layers.Conv1D(16, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(16, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(32, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(64, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(32, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(16, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(4, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Flatten())
        multi_chanel_model.add(tf.keras.layers.Dense(128, activation='swish'))
        #multi_chanel_model.add(Dropout(0.25))
        #multi_chanel_model.add(BatchNormalization())
        #multi_chanel_model.add(Dense(256, activation=tf.keras.layers.LeakyReLU()))
        #multi_chanel_model.add(Dropout(0.25))
        #multi_chanel_model.add(BatchNormalization())
        #multi_chanel_model.add(Dense(128, activation=tf.keras.layers.LeakyReLU()))
        #multi_chanel_model.add(Dropout(0.25))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.Dense(label_shape, activation='sigmoid'))

        out=multi_chanel_model(temporal_input)
        multi_chanel_model.summary()
        return out

    @staticmethod
    def model_builder_4(label_shape, temporal_input):
        # feature = tf.squeeze(tf.stack(input_list, axis=1), -4)
        temporal_input = tf.transpose(temporal_input, (0, 2, 1))
        multi_chanel_model = tf.keras.Sequential(name='total')
        multi_chanel_model.add(tf.keras.layers.InputLayer(temporal_input.shape[1:]))
        multi_chanel_model.add(
            tf.keras.layers.Conv1D(1, 3, padding='valid', activation='swish', input_shape=temporal_input.shape[1:]))
        multi_chanel_model.add(tf.keras.layers.Conv1D(16, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(16, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(32, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(64, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(32, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(16, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Conv1D(4, 3, activation='swish'))
        multi_chanel_model.add(tf.keras.layers.Dropout(0.5))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        multi_chanel_model.add(tf.keras.layers.Flatten())
        multi_chanel_model.add(tf.keras.layers.Dense(32, activation='swish'))
        # multi_chanel_model.add(Dropout(0.25))
        # multi_chanel_model.add(BatchNormalization())
        # multi_chanel_model.add(Dense(256, activation=tf.keras.layers.LeakyReLU()))
        # multi_chanel_model.add(Dropout(0.25))
        # multi_chanel_model.add(BatchNormalization())
        # multi_chanel_model.add(Dense(128, activation=tf.keras.layers.LeakyReLU()))
        # multi_chanel_model.add(Dropout(0.25))
        multi_chanel_model.add(tf.keras.layers.BatchNormalization())
        multi_chanel_model.add(tf.keras.layers.Dense(label_shape, activation='sigmoid'))

        out = multi_chanel_model(temporal_input)
        multi_chanel_model.summary()
        return out

    @staticmethod
    def model_builder_2(label_shape, temporal_input):
        # feature = tf.squeeze(tf.stack(input_list, axis=1), -4)
        temporal_input = tf.transpose(temporal_input, (0,2,1))
        length = temporal_input.shape[1]  # Number of Features (or length of the signal)
        model_width = 32  # Number of Filter or Kernel in the Input Layer (Power of 2 to avoid error)
        num_channel = 1  # Number of Input Channels
        problem_type = 'Regression'  # Regression or Classification
        output_number = label_shape

        #multi_chanel_model=DenseNet(length, num_channel, model_width, problem_type=problem_type, output_nums=output_number).DenseNet169()
        multi_chanel_model = Encoder(temporal_input,name='total')

        out = multi_chanel_model(temporal_input)
        multi_chanel_model.summary()
        return out

    @staticmethod
    def model_builder_3(label_shape, temporal_input):
        # feature = tf.squeeze(tf.stack(input_list, axis=1), -4)
        temporal_input = tf.transpose(temporal_input, (0, 2, 1))
        length = temporal_input.shape[1]  # Number of Features (or length of the signal)
        model_width = 32  # Number of Filter or Kernel in the Input Layer (Power of 2 to avoid error)
        num_channel = 1  # Number of Input Channels
        problem_type = 'Regression'  # Regression or Classification
        output_number = label_shape

        # multi_chanel_model=DenseNet(length, num_channel, model_width, problem_type=problem_type, output_nums=output_number).DenseNet169()
        multi_chanel_model = Encoder2(temporal_input, name='total')

        out = multi_chanel_model(temporal_input)
        multi_chanel_model.summary()
        return out

class Encoder(tf.keras.Model):
    ''' This the encoder part of VAE
    '''
    def __init__(self, temporal_input, hidden_dim=16, latent_dim=4,name='total'):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
        '''
        #super().__init__()

        conv=tf.keras.layers.Conv1D(16, kernel_size=7,strides=2,padding='same',activation='swish')
        #pool_1 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)
        layer = [conv]
        in_layer=tf.keras.Sequential(layer)

        denseblock1 = _make_dense_block(Dense_Block, True, 16)
        conv=tf.keras.layers.Conv1D(16, 5,strides=2, padding='same',activation='swish')
        layer = [conv,tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.25)]
        layer1 = tf.keras.Sequential(layer)

        denseblock2 = _make_dense_block(Dense_Block, True, 32)
        conv = tf.keras.layers.Conv1D(32, 5, strides=2, padding='same',activation='swish')
        layer = [conv,tf.keras.layers.BatchNormalization(),tf.keras.layers.AveragePooling1D(pool_size=7), tf.keras.layers.Dropout(0.25)]
        layer2 = tf.keras.Sequential(layer)

        denseblock3 = _make_dense_block(Dense_Block, True, 32)
        conv = tf.keras.layers.Conv1D(32, 5, strides=2, padding='same',activation='swish')
        layer = [conv,tf.keras.layers.BatchNormalization(),tf.keras.layers.AveragePooling1D(pool_size=5), tf.keras.layers.Dropout(0.25)]
        layer3 = tf.keras.Sequential(layer)

        denseblock4 = _make_dense_block(Dense_Block, True, 16)
        conv = tf.keras.layers.Conv1D(16, 5, strides=2, padding='same',activation='swish')
        layer = [conv,tf.keras.layers.BatchNormalization(),tf.keras.layers.AveragePooling1D(pool_size=3), tf.keras.layers.Dropout(0.25)]
        layer4 = tf.keras.Sequential(layer)

        denseblock5 = _make_dense_block(Dense_Block, True, 16)
        conv = tf.keras.layers.Conv1D(1, 5, strides=2, padding='same', activation='swish')
        layer = [conv, tf.keras.layers.BatchNormalization(), tf.keras.layers.Flatten(), tf.keras.layers.Dropout(0.25)]
        layer5 = tf.keras.Sequential(layer)


        encoder_hidden = tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim,activation='swish'),tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.25)])

        encoder_latent = tf.keras.Sequential([tf.keras.layers.Dense(latent_dim,activation='sigmoid'), tf.keras.layers.BatchNormalization()])

        out = in_layer(temporal_input)
        out = denseblock1(out)
        out = layer1(out)
        out = denseblock2(out)
        out = layer2(out)
        out = denseblock3(out)
        out = layer3(out)
        out = denseblock4(out)
        out = layer4(out)
        out = denseblock5(out)
        out = layer5(out)
        #out=tf.keras.layers.GlobalAveragePooling1D()(out)
        #out = out.view(-1, self.input_dim * 2)
        encoded_hidden = encoder_hidden(out)
        encoded_latent = encoder_latent(encoded_hidden)

        super(Encoder, self).__init__(inputs=temporal_input, outputs=encoded_latent,name=name)

class Encoder2(tf.keras.Model):
    ''' This the encoder part of VAE
    '''
    def __init__(self, temporal_input, hidden_dim=16, latent_dim=4,name='total'):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
        '''
        #super().__init__()

        conv=tf.keras.layers.Conv1D(16, kernel_size=3,strides=1,padding='same',activation='swish')
        #pool_1 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)
        layer = [conv]
        in_layer=tf.keras.Sequential(layer)

        denseblock1 = _make_dense_block(Dense_Block, True, 16)
        conv=tf.keras.layers.Conv1D(16, 3,strides=1, padding='same',activation='swish')
        layer = [conv,tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.25)]
        layer1 = tf.keras.Sequential(layer)

        denseblock2 = _make_dense_block(Dense_Block, True, 32)
        conv = tf.keras.layers.Conv1D(32, 3, strides=1, padding='same',activation='swish')
        layer = [conv,tf.keras.layers.BatchNormalization(),tf.keras.layers.AveragePooling1D(pool_size=2), tf.keras.layers.Dropout(0.25)]
        layer2 = tf.keras.Sequential(layer)

        denseblock3 = _make_dense_block(Dense_Block, True, 32)
        conv = tf.keras.layers.Conv1D(32, 3, strides=1, padding='same',activation='swish')
        layer = [conv,tf.keras.layers.BatchNormalization(),tf.keras.layers.AveragePooling1D(pool_size=2), tf.keras.layers.Dropout(0.25)]
        layer3 = tf.keras.Sequential(layer)

        denseblock4 = _make_dense_block(Dense_Block, True, 16)
        conv = tf.keras.layers.Conv1D(16, 3, strides=1, padding='same',activation='swish')
        layer = [conv,tf.keras.layers.BatchNormalization(),tf.keras.layers.AveragePooling1D(pool_size=2), tf.keras.layers.Dropout(0.25)]
        layer4 = tf.keras.Sequential(layer)

        denseblock5 = _make_dense_block(Dense_Block, True, 16)
        conv = tf.keras.layers.Conv1D(1, 3, strides=1, padding='same', activation='swish')
        layer = [conv, tf.keras.layers.BatchNormalization(), tf.keras.layers.Flatten(), tf.keras.layers.Dropout(0.25)]
        layer5 = tf.keras.Sequential(layer)


        encoder_hidden = tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim,activation='swish'),tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.25)])

        encoder_latent = tf.keras.Sequential([tf.keras.layers.Dense(latent_dim,activation='sigmoid'), tf.keras.layers.BatchNormalization()])

        out = in_layer(temporal_input)
        out = denseblock1(out)
        out = layer1(out)
        out = denseblock2(out)
        out = layer2(out)
        out = denseblock3(out)
        out = layer3(out)
        out = denseblock4(out)
        out = layer4(out)
        out = denseblock5(out)
        out = layer5(out)
        #out=tf.keras.layers.GlobalAveragePooling1D()(out)
        #out = out.view(-1, self.input_dim * 2)
        encoded_hidden = encoder_hidden(out)
        encoded_latent = encoder_latent(encoded_hidden)

        super(Encoder2, self).__init__(inputs=temporal_input, outputs=encoded_latent,name=name)


class Dense_Block(tf.keras.Model):
    def __init__(self, in_channels, is_encoder, growth_rate=16):
        super(Dense_Block, self).__init__()


        self.layer_0 = Dense_Block._layer_generator(growth_rate, is_encoder)
        self.layer_1 = Dense_Block._layer_generator(growth_rate, is_encoder)
        self.layer_2 = Dense_Block._layer_generator(growth_rate, is_encoder)
        #self.layer_3 = Dense_Block._layer_generator(growth_rate, is_encoder)
        #self.layer_4 = Dense_Block._layer_generator(growth_rate, is_encoder)


    def call(self,x):
        out_0 = self.layer_0(x)
        out_1 = self.layer_1(out_0)
        out_2 = self.layer_2(tf.concat([out_0, out_1], -1))
        #out_3 = self.layer_3(tf.concat([out_0, out_1, out_2], -1))
        #out_4 = self.layer_4(tf.concat([out_0, out_1, out_2, out_3], -1))
        out = tf.concat([out_0, out_1, out_2], -1)
        return out

    @staticmethod
    def _layer_generator(out_channels, is_encoder):
        if is_encoder:
            layer = [tf.keras.layers.Conv1D(out_channels,3,padding='same', activation='swish'), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.25)]
        else:
            layer = [tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
                     tf.keras.layers.Conv1D(out_channels,3,padding='same'),
                     ]
        return tf.keras.Sequential(layer)




def _make_dense_block(block, is_encoder, in_channels):
    layers = []
    layers.append(block(in_channels,is_encoder))
    return tf.keras.Sequential(layers)








