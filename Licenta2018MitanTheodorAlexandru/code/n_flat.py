import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from proj_utils import *
import random
import pickle
import time

old_progress = 0
def progress_bar(current, total, length):
    global old_progress
    progress = int(current / total * length)
    if progress != old_progress:
        old_progress = progress
        print('[' + progress * '=' + (length - progress) * ' ' + ']')

def img_flatten(img, label, classes, size=None, dmul=1, sort=True):
    x = np.array(img).reshape((32*32, 3))
    # print(x.shape)
    x = np.array(sorted(x, key=lambda pix: sum(pix))).flatten()
    y = np.zeros(len(classes))
    y[classes.index(label)] = 1
    return x, y

def gen_flat(imgfile='cifar_imgs.p', labelfile='cifar_labels.p', classes=[6, 8]):
    imgs = pickle.load(open(imgfile, 'rb'))
    labels = pickle.load(open(labelfile, 'rb'))
    imgs_sel = []
    labels_sel = []

    i = 0
    while i < len(labels):
        if labels[i] in classes:
            imgs_sel.append(imgs[i])
            labels_sel.append(labels[i])
        i += 1

    imgs, labels = imgs_sel, labels_sel
    print('selected samples:', len(imgs))
    total = len(imgs)
    x_arr = []
    y_arr = []
    for i in range(total):
        progress_bar(i, total, 20)
        x, y = img_flatten(imgs[i], labels[i], classes=classes, dmul=1)
        x_arr.append(x)
        y_arr.append(y)

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    
    print("x_arr.shape")
    print(x_arr.shape)
    print("y_arr.shape")
    print(y_arr.shape)
    return x_arr, y_arr

start = time.time()
x_arr, y_arr = gen_flat(imgfile='cifar_imgs.p', labelfile='cifar_labels.p', classes=[6, 8])
print('flatten time per:', (time.time() - start))


cutoff = (x_arr.shape[0] * 3) // 4
x_train = x_arr[0:cutoff]
y_train = y_arr[0:cutoff]
x_test = x_arr[cutoff:]
y_test = y_arr[cutoff:]
print('shape of x_train', np.shape(x_train))
print('shape of y_train', np.shape(y_train))
print('shape of x_test', np.shape(x_test))
print('shape of y_test', np.shape(y_test))

if True:
    # raise Exception('bop it stop it')
    # convert class vectors to binary class matrices
    if False:
        y_train = keras.utils.to_categorical(y_train, y_train.shape[1])
        y_test = keras.utils.to_categorical(y_test, y_train.shape[1])
    if False:
        # pickle data
        pack = (x_train, y_train, x_test, y_test)
        pickle.dump(pack, open('datapack', 'wb'))
    if False:
        # unpickle data
        x_train, y_train, x_test, y_test = pickle.load(open('datapack', 'rb'))
    
    batch_size = 1000
    epochs = 20
    model = Sequential()
    model.add(Dense(300, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    start = time.time()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    print('train time per:', (time.time() - start))

    start = time.time()
    score = model.evaluate(x_test, y_test, verbose=0)
    end = time.time()
    items = y_test.shape[0]
    print('time taken:', end - start)
    print('time taken per:', (end - start) / items)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print(x_train.shape)
    print(x_test.shape)
else:
    model = load_model('trained.h5')