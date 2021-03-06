from app.cnn.utils import *
from app.cnn.helper import *
from app.api.treelog import loog


def process_img(img_path, weight_name, input_maxsize=960):
    # K.get_session().close()
    # cfg = K.tf.ConfigProto()
    # cfg.gpu_options.allow_growth = True
    # cfg.gpu_options.per_process_gpu_memory_fraction = 0.2
    # K.set_session(K.tf.Session(config=cfg))
    # limit_mem()
    img = Image.open(img_path)
    img = img.convert('RGB')
    maxsize = (input_maxsize, input_maxsize)
    img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
    loog('np.array(img)', np.array(img).shape)
    img_arr = np.expand_dims(np.array(img), 0)
    inp, outp = get_model(img_arr)
    loog(inp, outp)
    model_sr = Model(inp, outp)
    path = '/home/lin/Downloads/imagenet/'
    weights_name = '{}.h5'.format(weight_name)
    # weights_name = 'top_model_in2_60000.h5'
    # weights_name = 'top_model_in2_7000.h5'
    # weights_name = 'top_model_in_25110_test.h5'
    # weights_name = 'top_model_one_epoch.h5'
    # weights_name = 'top_model_two_epoch.h5'
    # weights_name = 'top_model_half_epoch.h5'
    weights_path = path + weights_name
    model_sr.load_weights(weights_path)
    img_arr = model_sr.predict(img_arr)
    img_arr = img_arr[0].astype('uint8')
    img = to_pil_image(img_arr)
    img.save(img_path, quality=95)


def conv_block(x, filters, size, stride=(2, 2), mode='same', act=True):
    x = Conv2D(filters, (size, size), strides=stride, padding=mode)(x)
    # x = InstanceNormalization(axis=3)(x)
    x = BatchNormalization(axis=3)(x)
    return Activation('relu')(x) if act else x


def res_block(ip, nf=64):
    x = conv_block(ip, nf, 3, (1, 1))
    x = conv_block(x, nf, 3, (1, 1), act=False)
    return add([x, ip])


def up_block(x, filters, size):
    x = keras.layers.UpSampling2D()(x)
    x = Conv2D(filters, (size, size), strides=(1, 1), padding='same')(x)
    # x = InstanceNormalization(axis=3)(x)
    x = BatchNormalization(axis=3)(x)
    return Activation('relu')(x)


def get_model(arr):
    inp = Input(arr.shape[1:])
    x = conv_block(inp, 64, 9, (1, 1))
    for i in range(4): x = res_block(x)
    x = up_block(x, 64, 3)
    x = up_block(x, 64, 3)
    x = Conv2D(3, (9, 9), activation='tanh', strides=(1, 1), padding='same')(x)
    outp = Lambda(lambda x: (x + 1) * 127.5)(x)
    return inp, outp
