import importlib
import utils2
importlib.reload(utils2)
from utils2 import *
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from keras import metrics
from vgg16_avg import VGG16_Avg
from bcolz_array_iterator import BcolzArrayIterator

path = '/media/lin/4C3F9729770D0D36/imagenet/'
dpath = '/home/lin/Downloads/imagenet/'

rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]

deproc = lambda x, s: np.clip(x.reshape(s)[:, :, :, ::-1] + rn_mean, 0, 255)

arr_lr = bcolz.open(dpath + 'results/trn_resized_72_r.bc')
arr_hr = bcolz.open(path + 'results/trn_resized_288_r.bc')

parms = {'verbose': 0, 'callbacks': [TQDMNotebookCallback(leave_inner=True)]}


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


inp, outp = get_model(arr_lr)

shp = arr_hr.shape[1:]
vgg_inp = Input(shp)
vgg = VGG16(include_top=False, input_tensor=Lambda(preproc)(vgg_inp))
for l in vgg.layers: l.trainable = False


def get_outp(m, ln):
    return m.get_layer(f'block{ln}_conv2').output


vgg_content = Model(vgg_inp, [get_outp(vgg, o) for o in [1, 2, 3]])
vgg1 = vgg_content(vgg_inp)
vgg2 = vgg_content(outp)


def mean_sqr_b(diff):
    dims = list(range(1, K.ndim(diff)))
    return K.expand_dims(K.sqrt(K.mean(diff ** 2, dims)), 0)


w = [0.1, 0.8, 0.1]


def content_fn(x):
    res = 0;
    n = len(w)
    for i in range(n): res += mean_sqr_b(x[i] - x[i + n]) * w[i]
    return res


model_sr = Model([inp, vgg_inp], Lambda(content_fn)(vgg1 + vgg2))
model_sr.compile('adam', 'mae')


def train(bs, niter=10):
    targ = np.zeros((bs, 1))
    bc = BcolzArrayIterator(arr_hr, arr_lr, batch_size=bs)
    for i in range(niter):
        hr, lr = next(bc)
        model_sr.train_on_batch([lr[:bs], hr[:bs]], targ)


train(16, 80000)

model_sr.save_weights(dpath + 'model_sr_one_epoch.h5')
model_lr = Model(inp, outp)
model_lr.save_weights(dpath + 'model_lr_one_epoch.h5')

inp, outp = get_model(arr_hr)
model_hr = Model(inp, outp)
copy_weights(model_lr.layers, model_hr.layers)
model_hr.save_weights(dpath + 'model_hr_one_epoch.h5')

print ('success')
