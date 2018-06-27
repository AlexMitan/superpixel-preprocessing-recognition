import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from proj_utils import *
from imagesoup import ImageSoup
import random
import pickle
import time

def cifar_macro(img, label, classes, k=100, dmul=1, quantity=None, iter=10):
    xyrgb, w, h = convert(img, size=(32, 32))
    cent, lab = clusterise(xyrgb, k=k, dmul=dmul, iter=iter)
    q = int(k * k / 2) if quantity is None else quantity
    x = datagen(cent, 0, 1, quantity=q)[:, 0:8].flatten()
    # x = np.array(sorted(x[0:10], key=lambda pix: sum(pix))).flatten()
    y = np.zeros(num_classes)
    y[classes.index(label)] = 1
    return x, y

batch_size = 1000
epochs = 20
classes = [6, 8]
# classes = [0, 1]
num_classes = len(classes)
k = 30
quantity = 900


from_pickle = False
classes = [6, 8] # frog, ship
# classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# classes = [6, 9] # frog, truck
# classes = [3, 5] # cat, dog
# classes = [3, 8] # cat, ship
if not from_pickle:
    images = [[] for i in range(num_classes)]
    imgs = pickle.load(open('cifar_imgs.p', 'rb'))
    labels = pickle.load(open('cifar_labels.p', 'rb'))
    # imgs = pickle.load(open('sunmoon_imgs.p', 'rb'))
    # labels = pickle.load(open('sunmoon_labels.p', 'rb'))
    imgs_f = []
    labels_f = []
    i = 0

    while i < len(labels):
        if labels[i] in classes:
            imgs_f += [imgs[i]]
            labels_f += [labels[i]]
        i += 1

    imgs, labels = imgs_f, labels_f

    print('selected:', len(imgs_f))
    print(all([x in classes for x in labels_f]))

    x_arr = []
    y_arr = []

    start = time.time()
    total = len(imgs_f)
    print('starting preprocess for', total, 'images')
    for i in range(total):
        if i%100 is 0:
            print(i, '/', total)
        x, y = cifar_macro(imgs[i], labels[i], classes=classes, k=k, dmul=1, quantity=300, iter=4)
        x_arr.append(x)
        y_arr.append(y)
    time_taken = time.time() - start
    preprocess_time = time_taken / total
    print('time per preprocess', preprocess_time)
    print('total prep time', time_taken)

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)

    print("x_arr.shape")
    print(x_arr.shape)
    print("y_arr.shape")
    print(y_arr.shape)
    cutoff = (x_arr.shape[0] * 3) // 4
    x_train = x_arr[0:cutoff]
    y_train = y_arr[0:cutoff]
    x_test = x_arr[cutoff:]
    y_test = y_arr[cutoff:]
    pack = (x_train, y_train, x_test, y_test)
    pickle.dump(pack, open('relbatch datapack' + str(classes), 'wb'))
else:
    print('unpickling preprocessed input')
    # unpickle data
    x_train, y_train, x_test, y_test = pickle.load(open('relbatch datapack' + str(classes), 'rb'))


print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))

model = Sequential()
model.add(Dense(300, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
            optimizer=RMSprop(),
            metrics=['accuracy'])

start = time.time()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
end = time.time()
print('train time:', end - start)


start = time.time()
score = model.evaluate(x_test, y_test, verbose=0)
end = time.time()
items = y_test.shape[0]
print('time taken:', end - start)
print('time taken per:', (end - start) / items)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

show_confusion_matrix(y_test, model.predict(x_test), classes)