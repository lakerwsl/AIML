import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import math
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from sklearn.metrics import pairwise_distances

class rotate_resize_crop(layers.Layer):
    def __init__(self, minsize, maxsize, rotation):
        super(rotate_resize_crop, self).__init__()
        self.minsize = minsize
        self.maxsize = maxsize
        self.rotation = rotation

    def call(self, images):
      batch_size = tf.shape(images)[0]
      height = tf.shape(images)[1]
      width = tf.shape(images)[2]
      
      if self.rotation: 
        rand_angle = tf.random.uniform(shape=[], minval=-10, maxval=10, dtype=tf.float32)
        rotate = tfa.image.rotate(images,angles=rand_angle * math.pi / 180, interpolation="bilinear")
        rand_size = tf.random.uniform(shape=[], minval=self.minsize, maxval=self.maxsize, dtype=tf.int32)
        resize = tf.image.resize(rotate, (rand_size, rand_size))
        crop_resize = tf.image.random_crop(resize, (batch_size,height, width,1))
      else:
        rand_size = tf.random.uniform(shape=[], minval=self.minsize, maxval=self.maxsize, dtype=tf.int32)
        resize = tf.image.resize(images, (rand_size, rand_size))
        crop_resize = tf.image.random_crop(resize, (batch_size,height, width,1))

      return crop_resize

def augmenter(name, rotation):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1 / 255),
            rotate_resize_crop(minsize=29, maxsize=32, rotation=rotation),
        ],
        name=name,
    )
    
def n_augmenter(name):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1 / 255),
        ],
        name=name,
    )

def encoder(width):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(width, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(width, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(width, activation="relu"),
        ],
        name="encoder",
    )

class AIML(keras.Model):
    def __init__(
        self, temperature, tuneparameter, width,
    ):
        super(AIML, self).__init__()
        self.aiml_augmenter = augmenter(**aiml_augmenter)
        self.null_augmenter = n_augmenter(**null_augmenter)
        self.encoder = encoder(width)
        self.temperature = temperature
        self.tuneparameter = tuneparameter
        feature_dimensions = self.encoder.output_shape[1]

    def compile(self, aiml_optimizer, **kwargs):
        super(AIML, self).compile(**kwargs)
        self.aiml_optimizer = aiml_optimizer
        
    def aiml_loss(self, features_1, features_2, augmented_images_1, augmented_images_2):
        batch_size = tf.shape(features_1)[0]
        numfeature = tf.shape(features_1)[1]
        height = tf.shape(augmented_images_1)[1]
        width = tf.shape(augmented_images_1)[2]
        featureproduct = tf.matmul(features_1, features_2, transpose_a=True)
        
        indices = tf.range(start=0, limit=tf.shape(features_1)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        augmented_images_1_s = tf.gather(augmented_images_1, shuffled_indices)
        features_1_s = tf.gather(features_1, shuffled_indices)
        diffdist = tf.reduce_mean(tf.square(augmented_images_1_s - augmented_images_2), axis=(1,2,3))
        weight_1 = tf.exp(-diffdist/self.temperature)
        dist_s_1 = tf.reduce_mean(tf.square(features_1_s - features_2), axis=1)
        
        indices = tf.range(start=0, limit=tf.shape(features_2)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        augmented_images_2_s = tf.gather(augmented_images_2, shuffled_indices)
        features_2_s = tf.gather(features_2, shuffled_indices)
        diffdist = tf.reduce_mean(tf.square(augmented_images_2_s - augmented_images_1), axis=(1,2,3))
        weight_2 = tf.exp(-diffdist/self.temperature)
        dist_s_2 = tf.reduce_mean(tf.square(features_2_s - features_1), axis=1)
        
        unsupervised_loss = tf.reduce_mean(weight_1 * dist_s_1) + tf.reduce_mean(weight_2 * dist_s_2)
        selfsupervised_loss = tf.reduce_mean(tf.square(features_1-features_2))
        regularization_loss = tf.reduce_mean(tf.square(featureproduct - tf.eye(numfeature)))
        
        loss = self.tuneparameter[0] * unsupervised_loss / 2 + self.tuneparameter[1] * selfsupervised_loss + self.tuneparameter[2] * regularization_loss

        return loss
      
    def train_step(self, data):
        unlabeled_images, labels = data
        images = unlabeled_images
        augmented_images_1 = self.null_augmenter(images)
        augmented_images_2 = self.aiml_augmenter(images)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            aiml_loss = self.aiml_loss(features_1, features_2, augmented_images_1, augmented_images_2)
        gradients = tape.gradient(
            aiml_loss,
            self.encoder.trainable_weights,
        )
        self.aiml_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights,
            )
        )
        return {
            "aiml_loss": aiml_loss,
        }


def heat_kernel(dist, sigma):
  return np.exp(- (dist)/sigma)

def AILaplacian(data, temperature, width, ai_augmenter, n_aug):
  n=data.shape[0]
  aug_data = np.array(data).reshape(n,data.shape[1]*data.shape[2])
  for j in range(n_aug):
    aug_data = np.concatenate((aug_data,np.array(ai_augmenter(data)).reshape(n,data.shape[1]*data.shape[2])), axis=0)
  
  dist_matrix = pairwise_distances(aug_data)
  W=heat_kernel(dist_matrix, temperature)
  int_W=np.zeros((n, n))
  for i in range(n):
    irange = range(i, i+n*(n_aug+1), n)
    for j in range(n):
      jrange = range(j, j+n*(n_aug+1), n)
      int_W[i,j] = W[np.ix_(irange, jrange)].mean()
  Dsum = int_W.sum(axis=1)
  D = np.diag(Dsum)
  L=D-int_W
  D_tilde = np.diag(1/np.sqrt(D.diagonal()))
  L = np.matmul(D_tilde, np.matmul( L , D_tilde ))
  eigval, eigvec = np.linalg.eig(L)
  
  order = np.argsort(eigval)
  K_eigenvectors = eigvec[:, order[1:(width + 1)]]
#  KK_eigenvectors = np.zeros((n*(n_aug+1), width))
  for k in range(width):
    K_eigenvectors[:,k]=np.divide(K_eigenvectors[:,k],np.sqrt(Dsum))
#    KK_eigenvectors[:,k]=np.repeat(np.divide(K_eigenvectors[:,k],Dsum),n_aug+1)
  
  return(K_eigenvectors.real)
  

    
input_shape = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE
dataset_name = "mnist"
null_augmenter = {"name": "null_augmenter",}
shuffle_buffer = 5000
steps_per_epoch = 200
unlabelled_images = 60000
unlabeled_batch_size = unlabelled_images // steps_per_epoch
train_dataset = (tfds.load(dataset_name, split="train", as_supervised=True, shuffle_files=True)
      .shuffle(buffer_size=shuffle_buffer)
      .batch(unlabeled_batch_size, drop_remainder=True)
  )
width = 40
aiml_augmenter = {"name": "aiml_augmenter","rotation": True,}
model = AIML(temperature=0.03, tuneparameter = (1,100,200), width=width)
model.compile(aiml_optimizer=keras.optimizers.Adam())
pretrain_history = model.fit(train_dataset, epochs=25)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_tran = model.encoder(x_train)
x_test_tran = model.encoder(x_test)
np.savez('trandata_mnist_rotation.npz', x_train=x_train, x_train_tran=x_train_tran, y_train=y_train, x_test=x_test, x_test_tran=x_test_tran, y_test=y_test)


width = 40
aiml_augmenter = {"name": "aiml_augmenter","rotation": False}
model = AIML(temperature=0.03, tuneparameter = (1,100,200), width=width)
model.compile(aiml_optimizer=keras.optimizers.Adam())
pretrain_history = model.fit(train_dataset, epochs=25)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_tran = model.encoder(x_train)
x_test_tran = model.encoder(x_test)
np.savez('trandata_mnist.npz', x_train=x_train, x_train_tran=x_train_tran, y_train=y_train, x_test=x_test, x_test_tran=x_test_tran, y_test=y_test)

# print()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(f"Total training examples: {len(x_train)}")
# print(f"Total test examples: {len(x_test)}")
# #data=list(train_dataset.as_numpy_iterator())[1]
# #unlabeled_images, labels = data
# encore = model.encoder(x_train)
# plt.figure()
# imgplot = plt.imshow(encore)
# plt.show()
# 
# plt.figure()
# imgplot = plt.imshow(tf.reshape(images[4],(28,28)))
# plt.show()
# plt.figure()
# imgplot = plt.imshow(tf.reshape(rotate[4],(28,28)))
# plt.show()
# plt.figure()
# imgplot = plt.imshow(tf.reshape(my_images[4],(28,28)))
# plt.show()
