import importlib
from .utils2 import *
from .helper import *


def process_img(img_path):
    img = Image.open(img_path)
    img = np.expand_dims(np.array(img), 0)
    inp, outp = get_model(img)
    print(inp, outp)
    model_hr = Model(inp, outp)
    path = '/home/lin/Downloads/imagenet/'
    weights_path = path + 'top_model_one_epoch.h5'
    model_hr.load_weights(weights_path)
    img = model_hr.predict(img)
    img = to_pil_image(img)
    img.save(img_path, quality=95)



def conv_block(x, filters, size, stride=(2, 2), mode='same', act=True):
    x = Convolution2D(filters, size, size, subsample=stride, border_mode=mode)(x)
    x = BatchNormalization(mode=2)(x)
    return Activation('relu')(x) if act else x


def res_block(ip, nf=64):
    x = conv_block(ip, nf, 3, (1, 1))
    x = conv_block(x, nf, 3, (1, 1), act=False)
    return merge([x, ip], mode='sum')


def up_block(x, filters, size):
    x = keras.layers.UpSampling2D()(x)
    x = Convolution2D(filters, size, size, border_mode='same')(x)
    x = BatchNormalization(mode=2)(x)
    return Activation('relu')(x)


def get_model(arr):
    inp = Input(arr.shape[1:])
    x = conv_block(inp, 64, 9, (1, 1))
    for i in range(4): x = res_block(x)
    x = up_block(x, 64, 3)
    x = up_block(x, 64, 3)
    x = Convolution2D(3, 9, 9, activation='tanh', border_mode='same')(x)
    outp = Lambda(lambda x: (x + 1) * 127.5)(x)
    return inp, outp
