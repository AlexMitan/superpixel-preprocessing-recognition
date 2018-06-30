from PIL import Image
import numpy as np
import matplotlib as mpl
import random
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def remove_transparency(im, bg_colour=(255, 255, 255)):
    # credit to https://stackoverflow.com/questions/35859140/remove-transparency-alpha-from-any-image-using-pil#35859141
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im

def convert(img, shrink=None, size=None):
    if size is None:
        w, h = img.size
    else:
        w, h = size
    # shrink smallest side to a fixed number
    if shrink is not None:
        if max(w, h) > shrink:
            if w > h:
                w = round(w / h * shrink)
                h = shrink
            else:
                h = round(h / w * shrink)
                w = shrink
            img = img.resize((w, h))

    img_m = np.array(img)
    x, y = np.mgrid[:img_m.shape[0], :img_m.shape[1]]
    x = x / min(w, h)
    y = y / min(w, h)
    xyrgb = np.hstack([y.ravel()[:,None], 
                    x.ravel()[:,None],
                    img_m.reshape((-1,img_m.shape[-1])) / 255])
    return xyrgb.astype('float'), w, h


def clusterise(xyrgb, k, dmul=1, iter=None):
    '''xyrgb to centroids, labels'''  
    xyrgb[:, 0:2] *= dmul
    if iter is None:
        centroids, labels = kmeans2(xyrgb, k, minit='points')
    else:
        centroids, labels = kmeans2(xyrgb, k, minit='points', iter=iter)

    return centroids, labels

def reconstruct(w, h, centroids, labels):
    pixels = w * h
    # a tall array of r, g, b values with as many rows as pixels
    reconst = np.zeros((pixels, 3))
    for i in range(pixels):
        # get rgb value of the centroid the i-th pixel belongs to
        rgb = centroids[labels[i], 2:5] * 255
        # assign that rgb value to the pixel
        reconst[i] = rgb
    print('done')
    # reshape the matrix
    reconst = reconst.reshape((h, w, 3)).astype('uint8')

    return Image.fromarray(reconst)

def datagen(centroids, clss, n_classes, quantity=100):
    # prepare y
    # centroid list, class, n_classes -> relationships and onehot
    y = np.zeros((quantity, n_classes))
    y[:, clss] = 1

    x = np.zeros((quantity, 8))
    for i in range(quantity):
        first, second = random.choice(centroids), random.choice(centroids)
        first_xy, first_rgb = first[0:2], first[2:5]
        second_xy, second_rgb = second[0:2], second[2:5]
        
        xy_dist = np.linalg.norm(first_xy - second_xy)
        rgb_dist = np.linalg.norm(first_rgb - second_rgb)
        # dist = np.linalg.norm(first - second)
        # brightest first
        if np.sum(first_rgb) < np.sum(second_rgb):
            first_rgb, second_rgb = second_rgb, first_rgb
        x[i] = [*first_rgb, *second_rgb, xy_dist, rgb_dist]
    return np.hstack((x, y))

def data_macro(img, clss, n_classes, k=100, dmul=1, shrink=None, quantity=None):
    xyrgb, w, h = convert(img, shrink=shrink)
    cent, lab = clusterise(xyrgb, k=k, dmul=dmul)
    # reconstruct(w, h, cent, lab).save('reconst {}.jpg'.format(i * 10))
    q = int(k * k / 2) if quantity is None else quantity
    return datagen(cent, clss, n_classes, quantity=q)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def show_confusion_matrix(y_test, y_pred, classes):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    np.set_printoptions(precision=2)

    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] 										
    class_names = [class_names[x] for x in classes]
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                        title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                     title='Normalized confusion matrix')

    plt.show()

if __name__ == '__main__':
    for i in [3, 10, 30, 100, 300]:
        img = Image.open('mona lisa.png')
        print(np.shape(img))
        xyrgb, w, h = convert(img)
        cent, lab = clusterise(xyrgb, k=i, dmul=1)
        reconstruct(w, h, cent, lab).save('monareconst {}.jpg'.format(i))