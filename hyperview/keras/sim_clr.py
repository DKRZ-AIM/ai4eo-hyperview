import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D
from tensorflow.keras import backend as K
############################################

class SimCLR(tf.keras.Model):

    def __init__(self, base_model, input_shape):
        '''
        THIS IS THE CONSTRUCTOR OF SIM-CLR CLASS.
        :param base_model: backbone UNET model; it may not workwith some of the base models, model type 0, 5, and 6 are valid choices
        :param input_shape: (width,height,channel) shape of the input images
        :param order: it determines where to cut (from which layer order) UNET model for putting CNN + MLP header
        :return: returns nothing
        '''
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        header_shape = base_model.get_layer('total').output_shape[1:]
        header_input = base_model.get_layer('total').output


        model_h = Sequential([
            Input(shape=header_shape),
            Flatten(),
            Dense(4, activation='swish'),
            Lambda(lambda x: K.l2_normalize(x, axis=-1))

         ])


        composed_model = Model(inputs=[base_model.input], outputs=[model_h(header_input)])

        out_a = composed_model(input_a)
        out_b = composed_model(input_b)

        # CONCATENATED OUTPUT FOR SIM-CLR MODEL
        con_out = tf.keras.layers.Concatenate(axis=-1,name='contrastive')([out_a,out_b])

        # BINARY OUTPUT FOR SELF-SUPERVISED MODEL
        out = tf.math.subtract(out_a, out_b, name=None)
        out = Lambda(lambda out: tf.norm(out, ord='euclidean', keepdims=True, axis=-1))(out)
        binary =Dense(1, activation='sigmoid', name='binary')(out)


        super(SimCLR, self).__init__(inputs=[input_a, input_b], outputs=[con_out,binary])
        self.base_model = base_model
