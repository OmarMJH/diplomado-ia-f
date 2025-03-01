"""Modelos de redes neuronales varias."""
import attrs
import keras
# import tensorflow as tf
# from keras.models import Model
from keras.src.layers import (Input, Conv2D, MaxPooling2D, AveragePooling2D,
                              Flatten, GlobalAveragePooling2D, Dense, Dropout, concatenate)
from keras.src.models import Model


@attrs.define
class NNModels:
    """
    Clase para definir la arquitectura de los modelos de deep learning.

    Args:
        n_classes (int): Número de clases para la capa de salida.
        img_weight (int, default=150): Altura de la imagen.
        image_width (int, default=150): Ancho de la imagen.
        n_channels (int, default=3): Número de canales de la imagen.
    """
    n_classes: int = attrs.field(default=None)
    img_weight: int = attrs.field(default=150)
    image_width: int = attrs.field(default=150)
    n_channels: int = attrs.field(default=3)

    def __attrs_post_init__(self):

        print('tipo de modelo "alexnet" , "resnet50", "VGG16", "inceptionresnet", "efficientnet", o "googlenet" ')

    def ensamble_model(self):
        """Modelo ResNet50 basado en el modelo de Keras, cargado con los pesos de la base de datos
        de imagenet, sustituyendo la capa de salida a el numero de clases definido.

        Return:
            (Model): Modelo sin entrenar
        """
        # definir parametros de entrada y cargar modelo de keras.
        input_t = keras.Input(
            shape=(self.img_weight, self.image_width, self.n_channels))
        resnet = keras.applications.ResNet50(weights='imagenet',
                                             include_top=False,
                                             input_tensor=input_t)
        vgg16 = keras.applications.VGG16(weights='imagenet',
                                         include_top=False,
                                         input_tensor=input_t)
        efficient = keras.applications.EfficientNetB4(weights='imagenet',
                                                      include_top=False,
                                                      input_tensor=input_t)

        ensamble_input = [resnet.input, vgg16.input, efficient.input]

        # añadir capas de salida para las categorias de salida.
        resnet_model = resnet.output
        resnet_model = GlobalAveragePooling2D()(resnet_model)
        resnet_model = Dense(1024, activation='relu')(resnet_model)
        resnet_model = Dense(512, activation='relu')(resnet_model)
        prediction = Dense(self.n_classes, activation='softmax')(resnet_model)
        # model = Model(resnet.input, prediction)

        vgg_model = vgg16.output
        vgg_model = GlobalAveragePooling2D()(vgg_model)
        vgg_model = Dense(1024, activation='relu')(vgg_model)
        vgg_model = Dense(512, activation='relu')(vgg_model)
        prediction = Dense(self.n_classes, activation='softmax')(vgg_model)
        # model = Model(vgg16.input, prediction)

        ens_model = efficient.output
        ens_model = GlobalAveragePooling2D()(ens_model)
        ens_model = Dense(1024, activation='relu')(ens_model)
        ens_model = Dense(512, activation='relu')(ens_model)
        prediction = Dense(self.n_classes, activation='softmax')(ens_model)
        # model = Model(efficient.input, prediction)

        model = Model(ensamble_input, prediction)

        # (no) entrenar capas anteriores del modelo de keras
        for layer in resnet.layers:
            layer.trainable = False

        for layer in vgg16.layers:
            layer.trainable = False

        for layer in efficient.layers:
            layer.trainable = False

        return model

    def resnet50_model(self):
        """Modelo ResNet50 basado en el modelo de Keras, cargado con los pesos de la base de datos
        de imagenet, sustituyendo la capa de salida a el numero de clases definido.

        Return:
            (Model): Modelo sin entrenar
        """
        # definir parametros de entrada y cargar modelo de keras.
        input_t = keras.Input(
            shape=(self.img_weight, self.image_width, self.n_channels))
        resnet = keras.applications.ResNet50(weights='imagenet',
                                             include_top=False,
                                             input_tensor=input_t)

        # añadir capas de salida para las categorias de salida.
        resnet_model = resnet.output
        resnet_model = GlobalAveragePooling2D()(resnet_model)
        resnet_model = Dense(1024, activation='relu')(resnet_model)
        resnet_model = Dense(512, activation='relu')(resnet_model)
        prediction = Dense(self.n_classes, activation='softmax')(resnet_model)
        model = Model(resnet.input, prediction)

        # (no) entrenar capas anteriores del modelo de keras
        for layer in resnet.layers:
            layer.trainable = False

        return model

    def AlexNet_model(self):
        """Modelo AlexNet, definido con un input en base a los parametros de dimesionalidad de la
        clase y una salida con n_classes como categorias de salida.

        Return:
            (Model): Modelo sin entrenar
        """

        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                                input_shape=(self.img_weight, self.image_width, self.n_channels)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(
                1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(
                1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(
                1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(
                1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.n_classes, activation='softmax')
        ])

        return model

    def VGG16_model(self):
        """Modelo ResNet50 basado en el modelo de Keras, cargado con los pesos de la base de datos
        de imagenet, sustituyendo la capa de salida a el numero de clases definido.
        Return:
            (Model): Modelo sin entrenar
        """
        # definir parametros de entrada y cargar modelo de keras.
        input_t = keras.Input(
            shape=(self.img_weight, self.image_width, self.n_channels))
        vgg16 = keras.applications.VGG16(weights='imagenet',
                                         include_top=False,
                                         input_tensor=input_t)

        # añadir capas de salida para las categorias de salida.
        vgg_model = vgg16.output
        vgg_model = GlobalAveragePooling2D()(vgg_model)
        vgg_model = Dense(1024, activation='relu')(vgg_model)
        vgg_model = Dense(512, activation='relu')(vgg_model)
        prediction = Dense(self.n_classes, activation='softmax')(vgg_model)
        model = Model(vgg16.input, prediction)

        # (no) entrenar capas anteriores del modelo de keras
        for layer in vgg16.layers:
            layer.trainable = False

        return model

    def InceptionResnet_model(self):
        """Modelo inception resnet basado en el modelo de Keras, cargado con los pesos de la base
        de datos de imagenet, sustituyendo la capa de salida a el numero de clases definido.

        Return:
            (Model): Modelo sin entrenar
        """
        # definir parametros de entrada y cargar modelo de keras.
        input_t = keras.Input(
            shape=(self.img_weight, self.image_width, self.n_channels))
        iresnet = keras.applications.InceptionResNetV2(weights='imagenet',
                                                       include_top=False,
                                                       input_tensor=input_t)

        # añadir capas de salida para las categorias de salida.
        irn_model = iresnet.output
        irn_model = GlobalAveragePooling2D()(irn_model)
        irn_model = Dense(1024, activation='relu')(irn_model)
        irn_model = Dense(512, activation='relu')(irn_model)
        prediction = Dense(self.n_classes, activation='softmax')(irn_model)
        model = Model(iresnet.input, prediction)

        # (no) entrenar capas anteriores del modelo de keras
        for layer in iresnet.layers:
            layer.trainable = False

        return model

    def EfficientNet_model(self):
        """Modelo efficientnet basado en el modelo de Keras, cargado con los pesos de la base de
        datos de imagenet, sustituyendo la capa de salida a el numero de clases definido.

        Return:
            (Model): Modelo sin entrenar
        """
        # definir parametros de entrada y cargar modelo de keras.
        input_t = keras.Input(
            shape=(self.img_weight, self.image_width, self.n_channels))
        efficient = keras.applications.EfficientNetB4(weights='imagenet',
                                                      include_top=False,
                                                      input_tensor=input_t)

        # añadir capas de salida para las categorias de salida.
        ens_model = efficient.output
        ens_model = GlobalAveragePooling2D()(ens_model)
        ens_model = Dense(1024, activation='relu')(ens_model)
        ens_model = Dense(512, activation='relu')(ens_model)
        prediction = Dense(self.n_classes, activation='softmax')(ens_model)
        model = Model(efficient.input, prediction)

        # (no) entrenar capas anteriores del modelo de keras
        for layer in efficient.layers:
            layer.trainable = False

        return model

    def GoogLeNet_model(self):
        """Modelo GoogleNet, definido con un input en base a los parametros de dimensionalidad de la
        clase y una salida con n_classes como categorias de salida.

        Return:
            (Model): Modelo sin entrenar
        """
        # definir parametros de entrada y cargar modelo de keras.

        # input layer
        input_layer = Input(
            shape=(self.img_weight, self.image_width, self.n_channels))
        # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=2,
                   padding='valid', activation='relu')(input_layer)
        # max-pooling layer: pool_size = (3,3), strides = 2
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        # convolutional layer: filters = 64, strides = 1
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=1,
                   padding='same', activation='relu')(x)
        # convolutional layer: filters = 192, kernel_size = (3,3)
        x = Conv2D(filters=192, kernel_size=(3, 3),
                   padding='same', activation='relu')(x)
        # max-pooling layer: pool_size = (3,3), strides = 2
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        # 1st Inception block
        x = self.inception_block(x, f1=64, f2_conv1=96,
                                 f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32)
        # 2nd Inception block
        x = self.inception_block(x, f1=128, f2_conv1=128,
                                 f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64)
        # max-pooling layer: pool_size = (3,3), strides = 2
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        # 3rd Inception block
        x = self.inception_block(x, f1=192, f2_conv1=96,
                                 f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64)
        # Extra network 1:
        x1 = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
        x1 = Conv2D(filters=128, kernel_size=(1, 1),
                    padding='same', activation='relu')(x1)
        x1 = Flatten()(x1)
        x1 = Dense(1024, activation='relu')(x1)
        x1 = Dropout(0.7)(x1)
        # x1 = Dense(5, activation = 'softmax')(x1)
        x1 = Dense(self.n_classes, activation='softmax')(x1)
        # 4th Inception block
        x = self.inception_block(x, f1=160, f2_conv1=112,
                                 f2_conv3=224, f3_conv1=24, f3_conv5=64, f4=64)
        # 5th Inception block
        x = self.inception_block(x, f1=128, f2_conv1=128,
                                 f2_conv3=256, f3_conv1=24, f3_conv5=64, f4=64)
        # 6th Inception block
        x = self.inception_block(x, f1=112, f2_conv1=144,
                                 f2_conv3=288, f3_conv1=32, f3_conv5=64, f4=64)
        # Extra network 2:
        x2 = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
        x2 = Conv2D(filters=128, kernel_size=(1, 1),
                    padding='same', activation='relu')(x2)
        x2 = Flatten()(x2)
        x2 = Dense(1024, activation='relu')(x2)
        x2 = Dropout(0.7)(x2)
        # x2 = Dense(1000, activation = 'softmax')(x2)
        x2 = Dense(self.n_classes, activation='softmax')(x2)
        # 7th Inception block
        x = self.inception_block(x, f1=256, f2_conv1=160,
                                 f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)
        # max-pooling layer: pool_size = (3,3), strides = 2
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        # 8th Inception block
        x = self.inception_block(x, f1=256, f2_conv1=160,
                                 f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)
        # 9th Inception block
        x = self.inception_block(x, f1=384, f2_conv1=192,
                                 f2_conv3=384, f3_conv1=48, f3_conv5=128, f4=128)
        # Global Average pooling layer
        x = GlobalAveragePooling2D(name='GAPL')(x)
        # Dropoutlayer
        x = Dropout(0.4)(x)
        # output layer
        # x = Dense(1000, activation = 'softmax')(x)
        x = Dense(self.n_classes, activation='softmax')(x)
        # model
        model = Model(input_layer, [x, x1, x2], name='GoogLeNet')
        # model = Model(input_layer, x, name='GoogLeNet')

        return model

    @staticmethod
    def inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
        """
        Args:
            - f1: number of filters of the 1x1 convolutional layer in the first path
            - f2_conv1, f2_conv3: are number of filters corresponding to the 1x1 and
            3x3 convolutional layers in the second path
            - f3_conv1, f3_conv5: are the number of filters corresponding to the 1x1 and
            5x5 convolutional layer in the third path
            - f4: number of filters of the 1x1 convolutional layer in the fourth path
        """
        # 1st path:
        path1 = Conv2D(filters=f1, kernel_size=(1, 1),
                       padding='same', activation='relu')(input_layer)
        # 2nd path
        path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1),
                       padding='same', activation='relu')(input_layer)
        path2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3),
                       padding='same', activation='relu')(path2)
        # 3rd path
        path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1),
                       padding='same', activation='relu')(input_layer)
        path3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5),
                       padding='same', activation='relu')(path3)
        # 4th path
        path4 = MaxPooling2D((3, 3), strides=(1, 1),
                             padding='same')(input_layer)
        path4 = Conv2D(filters=f4, kernel_size=(1, 1),
                       padding='same', activation='relu')(path4)

        output_layer = concatenate([path1, path2, path3, path4], axis=-1)

        return output_layer
