'''
This method, while described in the written thesis, was de-emphasised due to its relative unstability
and unfeasability. Focus was then moved on to relbatch, and subsequently cellbatch.
The testing method has changed multiple times and is only meant for demonstrative purposes.
'''

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from proj_utils import *
from imagesoup import ImageSoup
import random
import pickle
import time 
batch_size = 1000
num_classes = 3
epochs = 5

k = 100
quantity = 2000

images = [[] for i in range(num_classes)]

def cifar_macro(img, label, num_classes=10, k=100, dmul=1, quantity=None):
    xyrgb, w, h = convert(img, size=(32, 32))
    cent, lab = clusterise(xyrgb, k=k, dmul=dmul)
    q = int(k * k / 2) if quantity is None else quantity
    return datagen(cent, label, num_classes, quantity=q)

imgs = pickle.load(open('cifar_imgs.p', 'rb'))
labels = pickle.load(open('cifar_labels.p', 'rb'))

imgs_f = []
labels_f = []
i = 0
while i < len(labels):
    if labels[i] < 3:
        imgs_f += [imgs[i]]
        labels_f += [labels[i]]
    i += 1

imgs, labels = imgs_f, labels_f

print('selected:', len(imgs_f))
print(all([x < 3 for x in labels_f]))

if True:
    data = None
    total = 100
    for i in range(total):
        sample = cifar_macro(imgs[i], labels[i], num_classes=num_classes, k=k, dmul=1, quantity=1000)
        if data is None:
            data = sample
        else:
            data = np.vstack((data, sample))
        print('done with sample', i, '/', total)
    total = data.shape[0]
    cutoff = int(total * 2 / 3)
    x_train = data[0:cutoff, 0:8]
    y_train = data[0:cutoff, 8:]
    x_test = data[cutoff:, 0:8]
    y_test = data[cutoff:, 8:]
    print(np.shape(x_train))
    print(np.shape(y_train))
    print(np.shape(x_test))
    print(np.shape(y_test))
    # raise Exception('bop it stop it')
    # convert class vectors to binary class matrices
    if False:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    if False:
        # pickle data
        pack = (x_train, y_train, x_test, y_test)
        pickle.dump(pack, open('datapack', 'wb'))
    if False:
        # unpickle data
        x_train, y_train, x_test, y_test = pickle.load(open('datapack', 'rb'))
    
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(8,)))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    start = time.time()
    score = model.evaluate(x_test, y_test, verbose=0)
    end = time.time()
    items = y_test.shape[0]
    print('time taken:', end - start)
    print('time taken per:', (end - start) / items)

    print(x_train.shape)
    print(x_test.shape)
else:
    model = load_model('trained.h5')

start = time.time()
correct = 0
wrong = 0
for i in range(3000, 3500):
    print('test', i - 3000)
    print('class', labels[i])
    sample = cifar_macro(imgs[i], labels[i], num_classes=num_classes, k=k, dmul=1, quantity=1000)

    
    out = model.predict(sample[:, 0:8])
    sums = np.sum(out, axis=0)
    if np.argmax(sums) == labels[i]:
        print('correct')
        correct += 1
    else:
        print('wrong')
        wrong += 1
end = time.time()
print('test time per:', (end - start) / 500)
print('correct: {}, wrong: {}, acc: {}'.format(correct, wrong, correct/(correct + wrong)))