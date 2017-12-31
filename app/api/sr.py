import importlib
from .utils2 import *
from .helper import *
from .treelog import loog


def process_img(img_path):
    img_arr = Image.open(img_path)
    loog(img_arr, 'img_arr1')
    img_arr = img_arr.astype('uint8')
    loog(img_arr, 'img_arr2')
    # img_arr = Image.open(img_path).resize((288, 288))
    img_arr = np.expand_dims(np.array(img_arr), 0)
    inp, outp = get_model(img_arr)
    print(inp, outp)
    model_hr = Model(inp, outp)
    path = '/home/lin/Downloads/imagenet/'
    weights_name = 'top_model_one_epoch.h5'
    weights_path = path + weights_name
    model_hr.load_weights(weights_path)
    img_arr = model_hr.predict(img_arr)
    img = to_pil_image(img_arr)
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
