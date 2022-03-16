import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from backbone_models.swin_transformer import SwinTransformer
from backbone_models.mobile_vit import MobileVit
from backbone_models.vit import ViT
from backbone_models.capsule_network import CapsNetBasic
from backbone_models.three_d_convolution_base import ThreeDCNN
from tensorflow.keras import activations
import os


class SpatioMultiChannellModel(tf.keras.Model):

    def __init__(self, model_type, channel_type,input_shape,label_shape,pretrained):
        super(SpatioMultiChannellModel, self).__init__()
        #https://keras.io/examples/vision/vivit/
        #https://www.philschmid.de/image-classification-huggingface-transformers-keras

        temporal_input = tf.keras.layers.Input(shape=input_shape)
        if channel_type==1:
            fet_out=SpatioMultiChannellModel._multi_channel_builder_1(model_type, pretrained, label_shape, temporal_input)
        elif channel_type == 2:
            fet_out = SpatioMultiChannellModel._multi_channel_builder_2(model_type, pretrained, label_shape,temporal_input)
        elif channel_type==3:
            fet_out = SpatioMultiChannellModel._multi_channel_builder_3(model_type, pretrained, label_shape,temporal_input)
        elif channel_type==4:
            fet_out=SpatioMultiChannellModel._multi_channel_builder_4(model_type, pretrained, label_shape,temporal_input)


        #input_layer = Input(shape=(16,))
        #reg_head=tf.keras.Sequential()
        #reg_head.add(InputLayer())
        #reg_head.add(Dropout(0.25))
        #reg_head.add(BatchNormalization())
        #reg_head.add(Dense(16, activation=tf.keras.layers.LeakyReLU()))
        #reg_head.add(Dropout(0.5))
        #reg_head.add(BatchNormalization())
        #reg_head.add(Dense(1, activation=tf.keras.layers.LeakyReLU()))

        #P_model = tf.keras.Model(input_layer, reg_head(input_layer),name='P')
        #K_model = tf.keras.Model(input_layer, reg_head(input_layer), name='K')
        #Mg_model = tf.keras.Model(input_layer, reg_head(input_layer), name='Mg')
        #pH_model = tf.keras.Model(input_layer, reg_head(input_layer), name='pH')


        #P_out = P_model(fet_out)
        #K_out = K_model(fet_out)
        #Mg_out = Mg_model(fet_out)
        #pH_out = pH_model(fet_out)

        [P_logit,K_logit,Mg_logit,pH_logit]=tf.unstack(fet_out, axis=-1)
        #fet_out = Layer(trainable=False,name='total')(fet_out)
        P_out = Activation(activation=activations.linear,name='P')(P_logit)
        K_out = Activation(activation=activations.linear, name='K')(K_logit)
        Mg_out = Activation(activation=activations.linear, name='Mg')(Mg_logit)
        pH_out = Activation(activation=activations.linear, name='pH')(pH_logit)


        super(SpatioMultiChannellModel, self).__init__(inputs=temporal_input, outputs=[fet_out, P_out, K_out, Mg_out, pH_out])

    @staticmethod
    def _multi_channel_builder_1(model_type,pretrained,label_shape, temporal_input):
        t_shape = temporal_input.shape
        input_list = tf.split(temporal_input, num_or_size_splits=int(t_shape[-1] / 3), axis=-1)
        feature = tf.squeeze(tf.stack(input_list, axis=1), -4)

        backbone = BackboneModel(model_type, feature.shape[2:], pretrained)
        # backbone.build((feature.shape[0],*feature.shape[2:]))
        # backbone.compile(run_eagerly=True)
        # backbone.summary()

        multi_chanel_model = tf.keras.Sequential(name='total')
        multi_chanel_model.add(TimeDistributed(backbone, input_shape=feature.shape[1:]))
        multi_chanel_model.add(Flatten())
        #multi_chanel_model.add(Dense(512, activation=tf.keras.layers.LeakyReLU()))
        #multi_chanel_model.add(Dropout(0.25))
        #multi_chanel_model.add(BatchNormalization())
        #multi_chanel_model.add(Dense(256, activation=tf.keras.layers.LeakyReLU()))
        #multi_chanel_model.add(Dropout(0.25))
        #multi_chanel_model.add(BatchNormalization())
        #multi_chanel_model.add(Dense(128, activation=tf.keras.layers.LeakyReLU()))
        #multi_chanel_model.add(Dropout(0.25))
        multi_chanel_model.add(BatchNormalization())
        multi_chanel_model.add(Dense(label_shape, activation='sigmoid'))

        out=multi_chanel_model(feature)
        return out

    @staticmethod
    def _multi_channel_builder_2(model_type,pretrained,label_shape, temporal_input):
        t_shape = temporal_input.shape
        input_list = tf.split(temporal_input, num_or_size_splits=int(t_shape[-1] / 3), axis=-1)
        out_list=[]
        for input in input_list:
            backbone = BackboneModel(model_type, input.shape[2:], pretrained)
            out_list.append(backbone(tf.squeeze(input, -4)))

        feature = tf.stack(out_list, axis=1)


        multi_chanel_model = tf.keras.Sequential(name='total')
        multi_chanel_model.add(Flatten())
        #multi_chanel_model.add(Dense(512, activation=tf.keras.layers.LeakyReLU()))
        #multi_chanel_model.add(Dropout(0.25))
        #multi_chanel_model.add(BatchNormalization())
        #multi_chanel_model.add(Dense(256, activation=tf.keras.layers.LeakyReLU()))
        #multi_chanel_model.add(Dropout(0.25))
        #multi_chanel_model.add(BatchNormalization())
        #multi_chanel_model.add(Dense(128, activation=tf.keras.layers.LeakyReLU()))
        #multi_chanel_model.add(Dropout(0.25))
        #multi_chanel_model.add(BatchNormalization())
        multi_chanel_model.add(Dense(label_shape, activation='sigmoid'))

        out=multi_chanel_model(feature)
        return out

    @staticmethod
    def _multi_channel_builder_3(model_type, pretrained, label_shape, temporal_input):
        t_shape = temporal_input.shape
        input=tf.squeeze(temporal_input,-4)
        #input_list = tf.split(temporal_input, num_or_size_splits=int(t_shape[-1] / 3), axis=-1)
        feature=CapsNetBasic(input,label_shape)



        multi_chanel_model = tf.keras.Sequential(name='total')
        multi_chanel_model.add(Flatten())
        # multi_chanel_model.add(Dense(512, activation=tf.keras.layers.LeakyReLU()))
        # multi_chanel_model.add(Dropout(0.25))
        # multi_chanel_model.add(BatchNormalization())
        # multi_chanel_model.add(Dense(256, activation=tf.keras.layers.LeakyReLU()))
        # multi_chanel_model.add(Dropout(0.25))
        # multi_chanel_model.add(BatchNormalization())
        # multi_chanel_model.add(Dense(128, activation=tf.keras.layers.LeakyReLU()))
        # multi_chanel_model.add(Dropout(0.25))
        # multi_chanel_model.add(BatchNormalization())
        multi_chanel_model.add(Dense(label_shape, activation='sigmoid'))

        out = multi_chanel_model(feature(input))
        return out

    @staticmethod
    def _multi_channel_builder_4(model_type, pretrained, label_shape, temporal_input):
        inp=tf.transpose(temporal_input, (0, 4, 2, 3, 1))
        model=ThreeDCNN(inp,label_shape)
        return model(inp)


class BackboneModel(tf.keras.Model):
        def __init__(self, model_type, input_shape,pretrained):

            inp = tf.keras.layers.Input(shape=input_shape)
            model=None
            weights = 'imagenet' if pretrained else None
            if model_type == 0:
                model=SwinTransformer('swin_tiny_224', num_classes=1000, include_top=False, pretrained=pretrained)

            if model_type == 1:
                if weights=='imagenet':
                    weights=os.path.join(os.getcwd(), 'models/weights_mobilenet_v3_small_224_1.0_float.h5')
                model=tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)

            if model_type == 2:
                if weights=='imagenet':
                    weights=os.path.join(os.getcwd(), 'models/weights_mobilenet_v3_large_224_1.0_float.h5')
                model=tf.keras.applications.MobileNetV3Large(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 3:
                if weights=='imagenet':
                    weights=os.path.join(os.getcwd(), 'models/efficientnetv2-s.h5')
                model=tf.keras.applications.EfficientNetV2S(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 4:
                if weights=='imagenet':
                    weights=os.path.join(os.getcwd(), 'models/evgg19_weights_tf_dim_ordering_tf_kernels.h5')
                model=tf.keras.applications.VGG19(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 5:
                if weights=='imagenet':
                    weights=os.path.join(os.getcwd(), 'models/xception_weights_tf_dim_ordering_tf_kernels.h5')
                model=tf.keras.applications.Xception(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 6:
                if weights=='imagenet':
                    weights=os.path.join(os.getcwd(), 'models/resnet50v2_weights_tf_dim_ordering_tf_kernels.h5')
                model=tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 7:
                model=MobileVit(input_shape=input_shape, include_top=False,classifier_activation=None)

            if model_type == 8:
                model = ViT(input_shape=input_shape, include_top=False, classifier_activation=None)


            single_channel_header = tf.keras.Sequential()
            #single_channel_header.add(GlobalAvgPool2D())
            single_channel_header.add(Flatten())
            single_channel_header.add(Dense(4, activation='sigmoid'))
            #single_channel_header.add(Dense(512, activation=tf.keras.layers.LeakyReLU()))
            #single_channel_header.add(Dropout(0.25))
            #single_channel_header.add(BatchNormalization())
            #single_channel_header.add(Dense(256, activation=tf.keras.layers.LeakyReLU()))
            #single_channel_header.add(Dropout(0.25))
            #single_channel_header.add(BatchNormalization())
            #single_channel_header.add(Dense(128, activation=tf.keras.layers.LeakyReLU()))
            #single_channel_header.add(Dropout(0.25))
            #single_channel_header.add(BatchNormalization())

            single_out = single_channel_header(model(inp))
            #backbone_with_head = tf.keras.Model(single_in, single_out)

            super(BackboneModel, self).__init__(inputs=inp, outputs=single_out)


        def compute_output_shape(self, input_shape):
            inp=tf.keras.layers.Input(shape=input_shape[1:])
            out=tf.keras.layers.Input(shape=4)
            #out=self.call(inp)
            return out.shape[:]
