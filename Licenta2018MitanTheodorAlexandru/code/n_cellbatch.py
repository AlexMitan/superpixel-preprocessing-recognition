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

def img_cellbatch(img, label, classes, size=None, k=100, dmul=1, sort=True, iter=None):
    xyrgb, w, h = convert(img, size=size)
    cent, lab = clusterise(xyrgb, k=k, dmul=dmul, iter=iter)
    if sort:
        cent = sorted(list(cent), key=lambda c: sum(c[2:5]))
    cent = np.array(cent).flatten()
    y = np.zeros(len(classes))
    y[classes.index(label)] = 1
    return cent, y

def gen_cellbatches(imgfile, labelfile, k, classes=[6, 8], iter=None, total=None, dmul=1):
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
    total = total if total is not None else len(imgs)
    print('selected samples:', total)

    start = time.time()
    
    print('started preprocessing for', total, 'images')
    x_arr = np.zeros((total, 5*k))
    y_arr = []
    for c in range(total):
        # print(c)
        progress_bar(c, total, 10)
        size = (imgs[c].shape[1], imgs[c].shape[0])
        x, y = img_cellbatch(imgs[c], labels[c], classes=classes, k=k, dmul=dmul, size=size, iter=iter)
        x_arr[c] = x[0:5*k]
        y_arr.append(y)

    y_arr = np.array(y_arr)
    time_taken = time.time() - start
    preprocess_time = time_taken / total
    print('time per preprocess', preprocess_time)
    print('total prep time', time_taken)
    
    print("x_arr.shape")
    print(x_arr.shape)
    print("y_arr.shape")
    print(y_arr.shape)
    return x_arr, y_arr

def make_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(x_test.shape[1], )))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])


    return model

from_pickle = True
classes = [6, 8] # frog, ship
# classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# classes = [6, 9] # frog, truck
# classes = [3, 5] # cat, dog
# classes = [3, 8] # cat, ship
if not from_pickle:
    print('generating input from raw data')
    x_arr, y_arr = gen_cellbatches(imgfile='cifar_imgs.p', labelfile='cifar_labels.p', classes=classes,
                                    k=30, iter=4, total=5000*len(classes), dmul=1)
    # x_arr, y_arr = gen_cellbatches(imgfile='sunmoon_imgs.p', labelfile='sunmoon_labels.p', classes=[0, 1],
    #                                 k=5, iter=4, total=400, dmul=1)

    cutoff = (x_arr.shape[0] * 3) // 4
    x_train = x_arr[0:cutoff]
    y_train = y_arr[0:cutoff]
    x_test = x_arr[cutoff:]
    y_test = y_arr[cutoff:]
    # pickle data
    pack = (x_train, y_train, x_test, y_test)
    pickle.dump(pack, open('cellbatch datapack' + str(classes), 'wb'))
else:
    print('unpickling preprocessed input')
    # unpickle data
    x_train, y_train, x_test, y_test = pickle.load(open('cellbatch datapack' + str(classes), 'rb'))

print('shape of x_train', np.shape(x_train))
print('shape of y_train', np.shape(y_train))
print('shape of x_test', np.shape(x_test))
print('shape of y_test', np.shape(y_test))

model = make_model(x_train, y_train, x_test, y_test)
start = time.time()
history = model.fit(x_train, y_train,
                    batch_size=1000,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_test, y_test))
end = time.time()
train_time = end - start

start = time.time()
score = model.evaluate(x_test, y_test, verbose=0)
end = time.time()
items = y_test.shape[0]
test_time =  (end - start) / items
# print('Test time taken per:', test_time)
# print('Test loss:', score[0])
print('Test accuracy:', score[1], ' ' * 200, score[1])
print('Train time:', train_time)

show_confusion_matrix(y_test, model.predict(x_test), classes)