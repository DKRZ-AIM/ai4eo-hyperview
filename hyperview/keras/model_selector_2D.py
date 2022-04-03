import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from backbone_models.swin_transformer import SwinTransformer
from backbone_models.mobile_vit import MobileVit,MobileVitC
from backbone_models.vit import ViT
from backbone_models.capsule_network import CapsNetBasic
from backbone_models.three_d_convolution_base import ThreeDCNN
from tensorflow.keras import activations
import os
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import math



class SpatioMultiChannellModel(tf.keras.Model):

    def __init__(self, model_type, channel_type,input_shape,label_shape,pretrained):
        #super(SpatioMultiChannellModel, self).__init__()
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
        elif channel_type==5:
            fet_out=SpatioMultiChannellModel._multi_channel_builder_5(model_type, pretrained, label_shape,temporal_input)
        elif channel_type==6:
            fet_out=SpatioMultiChannellModel._multi_channel_builder_6(model_type, pretrained, label_shape,temporal_input)

        elif channel_type==7:
            fet_out=SpatioMultiChannellModel._multi_channel_builder_7(model_type, pretrained, label_shape,temporal_input)


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
            #tf.keras.backend.clear_session()
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

    @staticmethod
    def _multi_channel_builder_5(model_type, pretrained, label_shape, temporal_input):

        input = tf.squeeze(temporal_input, -4)
        multi_chanel_model = tf.keras.Sequential()
        multi_chanel_model.add(Conv2D(filters=128, kernel_size=(1, 1)))
        multi_chanel_model.add(ECA(kernel=9))
        multi_chanel_model.add(Conv2D(filters=3, kernel_size=(1,1)))

        out = multi_chanel_model(input)

        backbone = BackboneModel(model_type, out.shape[1:], pretrained)

        out=backbone(out)
        out=Layer(name='total')(out)

        return out

    @staticmethod
    def _multi_channel_builder_6(model_type, pretrained, label_shape, temporal_input):

        input = tf.squeeze(temporal_input, -4)
        multi_chanel_model = tf.keras.Sequential()
        multi_chanel_model.add(Conv2D(filters=128, kernel_size=(1, 1)))
        multi_chanel_model.add(ECA(kernel=9))
        #multi_chanel_model.add(Conv2D(filters=3, kernel_size=(1, 1), activation='relu'))
        con1 = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')
        con2 = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')
        con3 = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')
        con4 = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')

        out = multi_chanel_model(input)
        out1 = con1(out)
        out2 = con2(out)
        out3 = con3(out)
        out4 = con4(out)

        backbone1 = BackboneModel(model_type, out1.shape[1:], pretrained,1)
        backbone2 = BackboneModel(model_type, out2.shape[1:], pretrained,1)
        backbone3 = BackboneModel(model_type, out3.shape[1:], pretrained,1)
        backbone4 = BackboneModel(model_type, out4.shape[1:], pretrained,1)

        out1 = backbone1(out1)
        out2 = backbone2(out2)
        out3 = backbone3(out3)
        out4 = backbone4(out4)

        out = tf.concat([out1,out2,out3,out4], axis=-1)



        out = Layer(name='total')(out)

        return out

    @staticmethod
    def _multi_channel_builder_7(model_type, pretrained, label_shape, temporal_input):

        input = tf.squeeze(temporal_input, -4)
        multi_chanel_model = tf.keras.Sequential()
        multi_chanel_model.add(Conv2D(filters=128, kernel_size=(1, 1)))
        multi_chanel_model.add(ECA(kernel=9,name='eca1'))
        multi_chanel_model.add(Conv2D(filters=16, kernel_size=(1, 1)))
        multi_chanel_model.add(ECA(kernel=3,name='eca2'))
        out = multi_chanel_model(input)

        backbone = BackboneModel(model_type, out.shape[1:], pretrained)

        out = backbone(out)
        out = Layer(name='total')(out)

        return out


class BackboneModel(tf.keras.Model):
        def __init__(self, model_type, input_shape,pretrained,out_shape=4):

            inp = tf.keras.layers.Input(shape=input_shape)
            model=None
            weights = 'imagenet' if pretrained else None
            if model_type == 0:
                model=SwinTransformer('swin_tiny_224', num_classes=1000, include_top=False, pretrained=pretrained)

            if model_type == 1:
                if weights=='imagenet':
                    weights=os.path.join(os.getcwd(), 'models/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5')
                model=tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)

            if model_type == 2:
                if weights=='imagenet':
                    weights=os.path.join(os.getcwd(), 'models/weights_mobilenet_v3_large_224_1.0_float_no_top_v2.h5')
                model=tf.keras.applications.MobileNetV3Large(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 3:
                if weights=='imagenet':
                    weights=os.path.join(os.getcwd(), 'models/efficientnetv2-s_notop.h5')
                model=tf.keras.applications.EfficientNetV2S(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 4:
                #if weights=='imagenet':
                    #weights=os.path.join(os.getcwd(), 'models/evgg19_weights_tf_dim_ordering_tf_kernels.h5')
                model=tf.keras.applications.VGG19(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 5:
                #if weights=='imagenet':
                    #weights=os.path.join(os.getcwd(), 'models/xception_weights_tf_dim_ordering_tf_kernels.h5')
                model=tf.keras.applications.Xception(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 6:
                #if weights=='imagenet':
                    #weights=os.path.join(os.getcwd(), 'models/resnet50v2_weights_tf_dim_ordering_tf_kernels.h5')
                model=tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=False,
                                                              classifier_activation=None,
                                                              weights=weights)
            if model_type == 7:
                model=MobileVit(input_shape=input_shape, include_top=False,classifier_activation=None)

            if model_type == 8:
                model = ViT(input_shape=input_shape, include_top=False, classifier_activation=None)

            if model_type == 9:
                model=MobileVitC(input_shape=input_shape, include_top=False,classifier_activation=None)



            single_channel_header = tf.keras.Sequential()
            #single_channel_header.add(GlobalAvgPool2D())
            single_channel_header.add(Flatten())
            single_channel_header.add(Dense(out_shape, activation='sigmoid'))
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



def get_gan_model(model_type, channel_type,input_shape,label_shape,pretrained):
    #gen_model = SpatioMultiChannellModel(model_type, channel_type,input_shape,label_shape,pretrained)
    #gen_model.build(tuple((None, *input_shape)))
    #gen_model.summary()
    #disc_model = DiscriminatorModel( input_shape, label_shape)
    #disc_model.build([tuple((None, *input_shape)),tuple((None, label_shape))])
    #disc_model.summary()
    gan=GAN(model_type, channel_type,input_shape,label_shape,pretrained)
    gan.build(tuple((None, *input_shape)))
    return gan


class GAN(tf.keras.Model):
    def __init__(self, model_type, channel_type,input_shape,label_shape,pretrained):
        super(GAN, self).__init__()
        self.gen_model=SpatioMultiChannellModel(model_type, channel_type,input_shape,label_shape,pretrained)
        #self.gen_model.build(tuple((None, *input_shape)))
        self.disc_model=DiscriminatorModel( input_shape, label_shape)
        #self.disc_model.build([tuple((None, *input_shape)), tuple((None, label_shape))])
        self.ema_gen_model = SpatioMultiChannellModel(model_type, channel_type,input_shape,label_shape,pretrained)
        #self.ema_gen_model.build(tuple((None, *input_shape)))
        #self.ema_gen_model.trainable=False
        self.ema = 0.9

    def predict(self, input_image):

       generated_images = self.ema_gen_model.predict(input_image)
       return generated_images

    def call(self, input_image, training=False):

        #input_image, target_image = data
        if training:
            generated_images = self.gen_model.call(input_image, training=training)
        else:
            generated_images = self.ema_gen_model.call(input_image,training=training)
        #gen_output = self.generate(data, training=training)
        #disc_real_output = self.disc_model([input_image, target_image], training=training)
        #disc_gene_output = self.disc_model([input_image, gen_output], training=training)
        return generated_images

    #def generate(self, data, training=False):
    #    input_image, target_image = data
    #    generated_images = self.gen_model(input_image, training=training)
    #    return generated_images


    def compile(self, g_optimizer,d_optimizer, gen_loss,disc_loss,core_metric,batch_size, **kwargs):
        super(GAN, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.gen_loss = gen_loss
        self.disc_loss=disc_loss
        self.core_metric=core_metric
        self.batch_size=batch_size

        self.generator_loss_tracker_train = tf.keras.metrics.Mean(name="loss")
        self.discriminator_loss_tracker_train = tf.keras.metrics.Mean(name="disc_loss")
        self.regression_loss_tracker_train = tf.keras.metrics.Mean(name="reg_loss")
        self.P_tracker_train = tf.keras.metrics.Mean(name="P_loss")
        self.K_tracker_train = tf.keras.metrics.Mean(name="K_loss")
        self.Mg_tracker_train = tf.keras.metrics.Mean(name="Mg_loss")
        self.pH_tracker_train= tf.keras.metrics.Mean(name="pH_loss")


    @property
    def train_metrics(self):
        return [
            self.generator_loss_tracker_train,
            self.discriminator_loss_tracker_train,
            self.regression_loss_tracker_train,
            self.P_tracker_train,
            self.K_tracker_train,
            self.Mg_tracker_train,
            self.pH_tracker_train,
        ]




    def test_step(self, data): #TODO https://keras.io/guides/customizing_what_happens_in_fit/
        input_image, target_image = data
        gen_output = self.gen_model(input_image, training=False)
        disc_real_output = self.disc_model([input_image, target_image], training=False)
        disc_gene_output = self.disc_model([input_image, gen_output[0]], training=False)

        gen_total_loss, gen_gan_loss, gen_l1_loss = self.gen_loss(disc_gene_output, gen_output[0], target_image,self.batch_size)
        disc_loss = self.disc_loss(disc_real_output, disc_gene_output, self.batch_size)
        p_m = self.core_metric(gen_output[0], target_image, 0)
        k_m = self.core_metric(gen_output[0], target_image, 1)
        mg_m = self.core_metric(gen_output[0], target_image, 2)
        ph_m = self.core_metric(gen_output[0], target_image, 3)

        self.generator_loss_tracker_train.update_state(gen_total_loss)
        self.regression_loss_tracker_train.update_state(gen_l1_loss)
        self.discriminator_loss_tracker_train.update_state(disc_loss)
        self.P_tracker_train.update_state(p_m)
        self.K_tracker_train.update_state(k_m)
        self.Mg_tracker_train.update_state(mg_m)
        self.pH_tracker_train.update_state(ph_m)

        return {m.name: m.result() for m in self.train_metrics[:]}


    def train_step(self, data):
        input_image, target_image = data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.gen_model(input_image,training=True)

            disc_real_output = self.disc_model([input_image, target_image], training=True)
            disc_gene_output = self.disc_model([input_image, gen_output[0]], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.gen_loss(disc_gene_output, gen_output[0], target_image,self.batch_size)
            disc_loss = self.disc_loss(disc_real_output, disc_gene_output,self.batch_size)
            p_m = self.core_metric(gen_output[0], target_image, 0)
            k_m = self.core_metric(gen_output[0], target_image, 1)
            mg_m = self.core_metric(gen_output[0], target_image, 2)
            ph_m = self.core_metric(gen_output[0], target_image, 3)

        generator_gradients = gen_tape.gradient(gen_total_loss,self.gen_model.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,self.disc_model.trainable_variables)

        self.g_optimizer.apply_gradients(zip(generator_gradients,self.gen_model.trainable_variables))
        self.d_optimizer.apply_gradients(zip(discriminator_gradients,self.disc_model.trainable_variables))

        self.generator_loss_tracker_train.update_state(gen_total_loss)
        self.regression_loss_tracker_train.update_state(gen_l1_loss)
        self.discriminator_loss_tracker_train.update_state(disc_loss)
        self.P_tracker_train.update_state(p_m)
        self.K_tracker_train.update_state(k_m)
        self.Mg_tracker_train.update_state(mg_m)
        self.pH_tracker_train.update_state(ph_m)

        # track the exponential moving average of the generator's weights to decrease
        # variance in the generation quality
        for weight, ema_weight in zip(self.gen_model.weights, self.ema_gen_model.weights):
            if weight.dtype in (np.int64, np.int32, np.int8):
                ema_weight.assign(weight)
            else:
                we=(self.ema * ema_weight + (1 - self.ema) * weight)
                ema_weight.assign(we)

        return {m.name: m.result() for m in self.train_metrics[:]}
        #disc_real_output = self.disc_model([input_image, target_image], training=True)
        #disc_gen_output = self.disc_model([input_image, gen_output], training=False)


        #labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        # Add random noise to the labels - important trick!
        #labels += 0.05 * tf.random.uniform(tf.shape(labels))
        #temp_in = tf.concat([input_image,input_image],axis=0)
        #temp_tar = tf.concat([target_image, gen_output], axis=0)
        # Train the discriminator
        #with tf.GradientTape(persistent=False) as tape:
        #    disc_output = self.disc_model([temp_in, temp_tar])
        #    d_loss = self.loss_fn(labels, disc_output)
        #grads = tape.gradient(d_loss, self.disc_model.trainable_weights)
        #self.d_optimizer.apply_gradients(zip(grads, self.disc_model.trainable_weights))

        #misleading_labels = tf.zeros((batch_size, 1))
        #with tf.GradientTape() as tape:
        #    gen_output = self.gen_model(input_image)
        #    predictions = self.disc_model([input_image, gen_output])
        #    g_loss = self.loss_fn(misleading_labels, predictions)
        #grads = tape.gradient(g_loss, self.generator.trainable_weights)
        #self.g_optimizer.apply_gradients(zip(grads, self.gen_model.trainable_weights))
        #return {"d_loss": d_loss, "g_loss": g_loss}

class DiscriminatorModel(tf.keras.Model):

        def __init__(self, image_stack_shape, target_shape):
            # super(SpatioTemporalModel, self).__init__()
            #in_shape=(28,28,1)
            init = RandomNormal(stddev=0.02)
            # source image input
            in_src_image = Input(shape=image_stack_shape)
            inp = tf.squeeze(in_src_image, 1)
            # label input
            in_label = Input(shape=(target_shape,))
            li = Embedding(target_shape, 150)(in_label)
            # scale up to image dimensions with linear activation
            n_nodes = image_stack_shape[1] * image_stack_shape[2]* image_stack_shape[3]/target_shape
            li = Dense(int(n_nodes))(li)
            # reshape to additional channel
            li = Reshape((image_stack_shape[1] , image_stack_shape[2], image_stack_shape[3]))(li)



            # concatenate images channel-wise
            merged = Concatenate()([inp, li])
            # C64
            d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
            d = LeakyReLU(alpha=0.2)(d)
            # C128
            d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            # C256
            d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            # C512
            d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            # second last output layer
            d = Conv2D(128, (4, 4), padding='same', kernel_initializer=init)(d)
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            # patch output
            d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
            d = Flatten()(d)
            d = Dense(1,activation='sigmoid')(d)

            super(DiscriminatorModel, self).__init__(inputs=[in_src_image, in_label], outputs=d)


# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Feb 03, 2021 
@file: DANet_attention3D.py
@desc: Dual attention network.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import tensorflow as tf


class Channel_attention(tf.keras.layers.Layer):
    """
    Channel attention module

    Fu, Jun, et al. "Dual attention network for scene segmentation."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint
        super(Channel_attention, self).__init__(**kwargs)

    def get_config(self):
        config = super(Channel_attention, self).get_config().copy()
        config.update({
            'gamma_initializer': self.gamma_initializer,
            'gamma_regularizer': self.gamma_regularizer,
            'gamma_constraint': self.gamma_constraint
        })
        return config

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        super(Channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4]))(inputs)
        proj_key = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        energy = tf.keras.backend.batch_dot(proj_query, proj_key)
        attention = tf.keras.activations.softmax(energy)

        outputs = tf.keras.backend.batch_dot(attention, proj_query)
        outputs = tf.keras.layers.Reshape((input_shape[1], input_shape[2], input_shape[3],
                                           input_shape[4]))(outputs)
        outputs = self.gamma * outputs + inputs

        return outputs


class ECA(tf.keras.layers.Layer):
    """ECA Conv layer.
    NOTE: This should be applied after a convolution operation.
    Shapes:
        INPUT: (B, C, H, W)
        OUPUT: (B, C_2, H, W)
    Attributes:
        filters (int): number of channels of input
        eca_k_size (int): kernel size for the 1D ECA layer
    """

    def __init__(
            self,
            gamma=2,
            b=2,
            kernel=None,
            **kwargs):

        super(ECA, self).__init__()

        self.kwargs = kwargs
        self.b = b
        self.gamma=gamma
        self.kernel=kernel

    def build(self, input_shapes):

        if self.kernel==None:
            t = int(abs(math.log(input_shapes[-1], 2) + self.b)/self.gamma)
            k = t if t % 2 else t + 1
        else:
            k=self.kernel

        self.eca_conv = Conv1D(
            filters=1,
            kernel_size=k,
            padding='same',
            use_bias=False)

    def get_config(self):
        config = super(ECA, self).get_config().copy()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):

        # (B, C, 1)
        attn = tf.math.reduce_mean(x, [-3, -2])[:, :,tf.newaxis]
        # (B, C, 1)
        attn = self.eca_conv(attn)

        # (B, 1, 1, C)
        attn=tf.transpose(attn, [0, 2, 1])
        attn = tf.expand_dims(attn, -3)

        # (B, 1, 1, C)
        attn = tf.math.sigmoid(attn)

        return x * attn


class ChannelGate(tf.keras.layers.Layer):
    """Apply Channelwise attention to input.
    Shapes:
        INPUT: (B, C, H, W)
        OUPUT: (B, C, H, W)
    Attributes:
        gate_channels (int): number of channels of input
        reduction_ratio (int): factor to reduce the channels in FF layer
        pool_types (list): list of pooling operations
    """

    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=['avg', 'max'],
        **kwargs):

        super(ChannelGate, self).__init__()

        all_pool_types = {'avg', 'max'}
        if not set(pool_types).issubset(all_pool_types):
            raise ValueError('The available pool types are: {}'.format(all_pool_types))

        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio
        self.pool_types = pool_types
        self.kwargs = kwargs

    def build(self, input_shape):
        hidden_units = self.gate_channels // self.reduction_ratio
        self.mlp = models.Sequential([
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(self.gate_channels, activation=None)
        ])


    def apply_pooling(self, inputs, pool_type):
        """Apply pooling then feed into ff.
        Args:
            inputs (tf.ten
        Returns:
            (tf.tensor) shape (B, C)
        """

        if pool_type == 'avg':
            pool = tf.math.reduce_mean(inputs, [2, 3])
        elif pool_type == 'max':
            pool = tf.math.reduce_max(inputs, [2, 3])

        channel_att = self.mlp(pool)
        return channel_att

    def call(self, inputs):
        pools = [self.apply_pooling(inputs, pool_type) \
            for pool_type in self.pool_types]

        # (B, C, 1, 1)
        attn = tf.math.sigmoid(tf.math.add_n(pools))[:, :, tf.newaxis, tf.newaxis]

        return attn * inputs




class Position_attention(tf.keras.layers.Layer):
    """
    Position attention module

    Fu, Jun, et al. "Dual attention network for scene segmentation."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(self,
                 ratio=8,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        self.ratio = ratio
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint
        super(Position_attention, self).__init__(**kwargs)

    def get_config(self):
        config = super(Position_attention, self).get_config().copy()
        config.update({
            'ratio': self.ratio,
            'gamma_initializer': self.gamma_initializer,
            'gamma_regularizer': self.gamma_regularizer,
            'gamma_constraint': self.gamma_constraint
        })
        return config

    def build(self, input_shape):
        super(Position_attention, self).build(input_shape)
        self.query_conv = tf.keras.layers.Conv3D(filters=input_shape[-1] // self.ratio,
                                                 kernel_size=(1, 1, 1), use_bias=False,
                                                 kernel_initializer='he_normal')
        self.key_conv = tf.keras.layers.Conv3D(filters=input_shape[-1] // self.ratio,
                                               kernel_size=(1, 1, 1), use_bias=False,
                                               kernel_initializer='he_normal')
        self.value_conv = tf.keras.layers.Conv3D(filters=input_shape[-1], kernel_size=(1, 1, 1),
                                                 use_bias=False, kernel_initializer='he_normal')
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4] // self.ratio))(self.query_conv(inputs))
        proj_query = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        proj_key = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                            input_shape[4] // self.ratio))(self.key_conv(inputs))
        energy = tf.keras.backend.batch_dot(proj_key, proj_query)
        attention = tf.keras.activations.softmax(energy)

        proj_value = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4]))(self.value_conv(inputs))

        outputs = tf.keras.backend.batch_dot(attention, proj_value)
        outputs = tf.keras.layers.Reshape((input_shape[1], input_shape[2], input_shape[3],
                                           input_shape[4]))(outputs)
        outputs = self.gamma * outputs + inputs

        return outputs


