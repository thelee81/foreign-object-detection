import tensorflow as tf

from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers

import cv2
import numpy as np
import matplotlib.pyplot as plt



class AbstractNetwork():
    def __init__(self, input_shape, num_filters):
        self.input_shape = input_shape
        self.num_filters = num_filters

class Autoencoder(AbstractNetwork):
    def __init__(self, input_shape=(None, None, 3), num_filters=(16, 32, 64, 128, 256)):
        super().__init__(input_shape, num_filters)

    def build(self, activation="relu", padding="same"):
        
            # Encoder
            input = layers.Input(shape=self.input_shape)
            
            x = layers.Conv2D(self.num_filters[0], kernel_size=(7, 7), padding='same', strides=(1, 1))(input)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

            x = layers.Conv2D(self.num_filters[1], kernel_size=(5, 5), padding='same', strides=(1, 1))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
                
            x = layers.Conv2D(self.num_filters[2], kernel_size=(5, 5), padding='same', strides=(1, 1))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)    

            x = layers.Conv2D(self.num_filters[3], kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

            x = layers.Conv2D(self.num_filters[4], kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)


            # Bottleneck
            encoded = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

            # Decoder
            x = layers.Conv2DTranspose(self.num_filters[3], kernel_size=(3, 3), strides=(2, 2), padding='same')(encoded)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)

            x = layers.Conv2DTranspose(self.num_filters[2], kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)

            x = layers.Conv2DTranspose(self.num_filters[1], kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)

            x = layers.Conv2DTranspose(self.num_filters[0], kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)

            output = layers.Conv2DTranspose(self.input_shape[-1], kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
            output = layers.Activation("sigmoid")(output)

            self.model = Model(input, output)

            return self.model


if __name__ == "__main__":
    nn = Autoencoder(input_shape=(2048, 2048, 3))
    model = nn.build()

    print(model.summary())