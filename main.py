# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import urllib
import urllib.request
from zipfile import ZipFile
import cv2
import numpy as np
import matplotlib.pyplot as plt

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

np.set_printoptions(linewidth=200)


def download_file():
    if not os.path.isfile(FILE):
        print(f'Downloading {URL} and saving as {FILE}')
        urllib.request.urlretrieve(URL, FILE)


def unzip_file():
    print("Unzipping images...")
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)
    print('DONE')


def load_image_date():
    image_data = cv2.imread('fashion_mnist_images/train/4/0011.png', cv2.IMREAD_UNCHANGED)
    plt.imshow(image_data, cmap='gray')
    plt.show()


def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # download_file()
    # unzip_file()
    # load_image_date()
    print("here")
    X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
    X = (X.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    X = X.reshape(X.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print('X.shape[0]', X.shape[0])
    keys = np.array(range(X.shape[0]))
    print("keys", keys)
    np.random.shuffle(keys)
    print('shuffled keys', keys[:10])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
