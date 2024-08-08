import numpy as np
import tensorflow as tf
import os


def load_data():
    data_dir = '/Users/shuleiwang/Library/CloudStorage/Dropbox/Data/ImageNet/'
    data_path_train_1 = os.path.join(data_dir + 'Imagenet32_train_npz/', 'train_data_batch_')
    data_path_val = os.path.join(data_dir + 'Imagenet32_val_npz/', 'val_data.npz')

    img_size = 32
    img_size2 = img_size * img_size

    idx = 1
    with np.load(data_path_train_1 + str(idx) + '.npz') as data:
        x = data['data']
        y = data['labels']

    data_size = x.shape[0]
    X_train = x[0:data_size, :]
    Y_train = y[0:data_size]

    for idx in range(9):
        with np.load(data_path_train_1 + str(idx + 2) + '.npz') as data:
            x = data['data']
            y = data['labels']

        data_size = x.shape[0]
        X_train = np.concatenate((X_train, x[0:data_size, :]), axis=0)
        Y_train = np.concatenate((Y_train, y[0:data_size]), axis=0)

    print(X_train.shape)
    print(Y_train.shape)

    X_train = np.dstack((X_train[:, :img_size2], X_train[:, img_size2:2 * img_size2], X_train[:, 2 * img_size2:]))
    X_train = X_train.reshape((X_train.shape[0], img_size, img_size, 3))

    print(X_train.shape)
    print(Y_train.shape)

    with np.load(data_path_val) as data:
        x = data['data']
        y = data['labels']

    data_size = x.shape[0]
    x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    X_val = x[0:data_size, :, :, :]
    Y_val = y[0:data_size]

    print(X_val.shape)
    print(Y_val.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

    return train_dataset, val_dataset

