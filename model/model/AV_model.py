from keras.models import Sequential
from keras.layers import Input, Dense, Convolution2D,Bidirectional, concatenate
from keras.layers import Flatten, BatchNormalization, ReLU, Reshape, Lambda, TimeDistributed,Activation

from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.initializers import he_normal, glorot_uniform
import tensorflow as tf



def AV_model(people_num=2):
    def UpSampling2DBilinear(size):
        return Lambda(lambda x: tf.image.resize(x, size, method=tf.image.ResizeMethod.BILINEAR))

    def sliced(x, index):
        return x[:, :, :, index]

    # --------------------------- AS start ---------------------------
    audio_input = Input(shape=(298, 257, 2))
    AS_model = Sequential()
    print('as_0:', audio_input.shape)
    AS_model.add(Convolution2D(96, kernel_size=(1, 7), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv1',input_shape=audio_input.shape))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv1 = Convolution2D(96, kernel_size=(1, 7), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv1')(audio_input)
    # as_conv1 = BatchNormalization()(as_conv1)
    # as_conv1 = ReLU()(as_conv1)
    print('as_1:', model.layers[0].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    
    # as_conv2 = Convolution2D(96, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv2')(as_conv1)
    # as_conv2 = BatchNormalization()(as_conv2)
    # as_conv2 = ReLU()(as_conv2)
    print('as_2:', model.layers[1].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv3 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv3')(as_conv2)
    # as_conv3 = BatchNormalization()(as_conv3)
    # as_conv3 = ReLU()(as_conv3)
    print('as_3:', model.layers[2].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 1)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv4 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='as_conv4')(as_conv3)
    # as_conv4 = BatchNormalization()(as_conv4)
    # as_conv4 = ReLU()(as_conv4)
    print('as_4:', model.layers[3].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 1)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv5 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='as_conv5')(as_conv4)
    # as_conv5 = BatchNormalization()(as_conv5)
    # as_conv5 = ReLU()(as_conv5)
    print('as_5:', model.layers[4].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 1)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv6 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='as_conv6')(as_conv5)
    # as_conv6 = BatchNormalization()(as_conv6)
    # as_conv6 = ReLU()(as_conv6)
    print('as_6:', model.layers[5].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 1)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv7 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 1), name='as_conv7')(as_conv6)
    # as_conv7 = BatchNormalization()(as_conv7)
    # as_conv7 = ReLU()(as_conv7)
    print('as_7:', model.layers[6].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 1)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))    
    # as_conv8 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 1), name='as_conv8')(as_conv7)
    # as_conv8 = BatchNormalization()(as_conv8)
    # as_conv8 = ReLU()(as_conv8)
    print('as_8:', model.layers[7].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv9 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv9')(as_conv8)
    # as_conv9 = BatchNormalization()(as_conv9)
    # as_conv9 = ReLU()(as_conv9)
    print('as_9:', model.layers[8].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 2)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv10 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 2), name='as_conv10')(as_conv9)
    # as_conv10 = BatchNormalization()(as_conv10)
    # as_conv10 = ReLU()(as_conv10)
    print('as_10:', model.layers[9].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 4)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv11 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 4), name='as_conv11')(as_conv10)
    # as_conv11 = BatchNormalization()(as_conv11)
    # as_conv11 = ReLU()(as_conv11)
    print('as_11:', model.layers[10].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 8)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv12 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 8), name='as_conv12')(as_conv11)
    # as_conv12 = BatchNormalization()(as_conv12)
    # as_conv12 = ReLU()(as_conv12)
    print('as_12:', model.layers[11].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 16)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv13 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 16), name='as_conv13')(as_conv12)
    # as_conv13 = BatchNormalization()(as_conv13)
    # as_conv13 = ReLU()(as_conv13)
    print('as_13:', model.layers[12].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 32)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv14 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 32), name='as_conv14')(as_conv13)
    # as_conv14 = BatchNormalization()(as_conv14)
    # as_conv14 = ReLU()(as_conv14)
    print('as_14:', model.layers[13].output_shape)

    AS_model.add(Convolution2D(96, kernel_size=(1, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1)))
    AS_model.add(BatchNormalization())
    AS_model.add(Activation('relu'))
    # as_conv15 = Convolution2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv15')(as_conv14)
    # as_conv15 = BatchNormalization()(as_conv15)
    # as_conv15 = ReLU()(as_conv15)
    print('as_15:', model.layers[14].output_shape)

    AS_out = Reshape((298, 8 * 257))
    print('AS_out:', AS_out.shape)
    # --------------------------- AS end ---------------------------

    # --------------------------- VS_model start ---------------------------
    VS_model = Sequential()
    VS_model.add(Convolution2D(256, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='vs_conv1'))
    VS_model.add(BatchNormalization())
    VS_model.add(Activation('relu'))
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='vs_conv2'))
    VS_model.add(BatchNormalization())
    VS_model.add(Activation('relu'))
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='vs_conv3'))
    VS_model.add(BatchNormalization())
    VS_model.add(Activation('relu'))
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='vs_conv4'))
    VS_model.add(BatchNormalization())
    VS_model.add(Activation('relu'))
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='vs_conv5'))
    VS_model.add(BatchNormalization())
    VS_model.add(Activation('relu'))
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(16, 1), name='vs_conv6'))
    VS_model.add(BatchNormalization())
    VS_model.add(Activation('relu'))
    VS_model.add(Reshape((75, 256, 1)))
    VS_model.add(UpSampling2DBilinear((298, 256)))
    VS_model.add(Reshape((298, 256)))
    # --------------------------- VS_model end ---------------------------

    video_input = Input(shape=(75, 1, 1792, people_num))
    AVfusion_list = [AS_out]
    for i in range(people_num):
        single_input = Lambda(sliced, arguments={'index': i})(video_input)
        VS_out = VS_model(single_input)
        AVfusion_list.append(VS_out)

    AVfusion = concatenate(AVfusion_list, axis=2)
    AVfusion = TimeDistributed(Flatten())(AVfusion)
    print('AVfusion:', AVfusion.shape)

    lstm = Bidirectional(LSTM(400, input_shape=(298, 8 * 257), return_sequences=True), merge_mode='sum')(AVfusion)
    print('lstm:', lstm.shape)

    fc1 = Dense(600, name="fc1", activation='relu', kernel_initializer=he_normal(seed=27))(lstm)
    print('fc1:', fc1.shape)
    fc2 = Dense(600, name="fc2", activation='relu', kernel_initializer=he_normal(seed=42))(fc1)
    print('fc2:', fc2.shape)
    fc3 = Dense(600, name="fc3", activation='relu', kernel_initializer=he_normal(seed=65))(fc2)
    print('fc3:', fc3.shape)

    complex_mask = Dense(257 * 2 * people_num, name="complex_mask", kernel_initializer=glorot_uniform(seed=87))(fc3)
    print('complex_mask:', complex_mask.shape)

    complex_mask_out = Reshape((298, 257, 2, people_num))(complex_mask)
    print('complex_mask_out:', complex_mask_out.shape)

    AV_model = Model(inputs=[audio_input, video_input], outputs=complex_mask_out)

    # # compile AV_model
    # AV_model.compile(optimizer='adam', loss='mse')

    return AV_model
