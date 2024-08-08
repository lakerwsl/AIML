import tensorflow as tf
import tensorflow_datasets as tfds
import Loading_ImageNet as lin
import resnet20
import matplotlib.pyplot as plt
import Methods_ImageNet as methods
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import random

import os

shuffle_buffer = 50000
batch_size = 2000
epochs = 100
steps_per_epoch = 1281167 // batch_size
total_steps = steps_per_epoch * epochs
width = 2048
kernels = 128
size = 32
input_shape = (size, size, 3)
null_augmenter = {"name": "null_augmenter", }
aiml_augmenter = {"name": "aiml_augmenter", }
c_augmenter = {"name": "classification_augmenter",}
img_size = 32
img_size2 = img_size * img_size
AUTO = tf.data.AUTOTUNE

class RandomResizedCrop(layers.Layer):
    def __init__(self, scale, ratio, **kwargs):
        super().__init__(**kwargs)
        # area-range of the cropped part: (min area, max area), uniform sampling
        self.scale = scale
        # aspect-ratio-range of the cropped part: (log min ratio, log max ratio), log-uniform sampling
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]
            height = tf.shape(images)[1]
            width = tf.shape(images)[2]

            # independently sampled scales and ratios for every image in the batch
            random_scales = tf.random.uniform(
                (batch_size,), self.scale[0], self.scale[1]
            )
            random_ratios = tf.exp(
                tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1])
            )

            # corresponding height and widths, clipped to fit in the image
            new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
            new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)

            # random anchors for the crop bounding boxes
            height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
            width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

            # assemble bounding boxes and crop
            bounding_boxes = tf.stack(
                [
                    height_offsets,
                    width_offsets,
                    height_offsets + new_heights,
                    width_offsets + new_widths,
                ],
                axis=1,
            )
            images = tf.image.crop_and_resize(
                images, bounding_boxes, tf.range(batch_size), (height, width)
            )

        return images


# distorts the color distibutions of images
class RandomColorJitter(layers.Layer):
    def __init__(self, brightness, contrast, saturation, hue, **kwargs):
        super().__init__(**kwargs)

        # color jitter ranges: (min jitter strength, max jitter strength)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        # list of applicable color augmentations
        self.color_augmentations = [
            self.random_brightness,
            self.random_contrast,
            self.random_saturation,
            self.random_hue,
        ]

        # the tf.image.random_[brightness, contrast, saturation, hue] operations
        # cannot be used here, as they transform a batch of images in the same way

    def blend(self, images_1, images_2, ratios):
        # linear interpolation between two images, with values clipped to the valid range
        return tf.clip_by_value(ratios * images_1 + (1.0 - ratios) * images_2, 0, 1)

    def random_brightness(self, images):
        # random interpolation/extrapolation between the image and darkness
        return self.blend(
            images,
            0,
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.brightness, 1 + self.brightness
            ),
        )

    def random_contrast(self, images):
        # random interpolation/extrapolation between the image and its mean intensity value
        mean = tf.reduce_mean(
            tf.image.rgb_to_grayscale(images), axis=(1, 2), keepdims=True
        )
        return self.blend(
            images,
            mean,
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.contrast, 1 + self.contrast
            ),
        )

    def random_saturation(self, images):
        # random interpolation/extrapolation between the image and its grayscale counterpart
        return self.blend(
            images,
            tf.image.rgb_to_grayscale(images),
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.saturation, 1 + self.saturation
            ),
        )

    def random_hue(self, images):
        # random shift in hue in hsv colorspace
        images = tf.image.rgb_to_hsv(images)
        images += tf.random.uniform(
            (tf.shape(images)[0], 1, 1, 3), (-self.hue, 0, 0), (self.hue, 0, 0)
        )
        # tf.math.floormod(images, 1.0) should be used here, however in introduces artifacts
        images = tf.where(images < 0.0, images + 1.0, images)
        images = tf.where(images > 1.0, images - 1.0, images)
        images = tf.image.hsv_to_rgb(images)
        return images

    def call(self, images, training=True):
        if training:
            # applies color augmentations in random order
            for color_augmentation in random.sample(self.color_augmentations, 4):
                images = color_augmentation(images)
        return images

def encoder(width):
    return resnet20.get_network(hidden_dim=width, use_pred=False, return_before_head=False)

def augmenter(name):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            RandomResizedCrop(scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)),
            RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        ],
        name=name,
    )


def n_augmenter(name):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            RandomResizedCrop(scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)),
            RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        ],
        name=name,
    )
def classification_augmenter(name):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            RandomResizedCrop(scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
            RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ],
        name=name,
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
        batch_size = tf.shape(features_1, out_type=tf.float32)[0]
        numfeature = tf.shape(features_1)[1]
        height = tf.shape(augmented_images_1)[1]
        width = tf.shape(augmented_images_1)[2]
        featureproduct = (tf.matmul(features_1, features_2, transpose_a=True) / batch_size)

        indices = tf.range(start=0, limit=tf.shape(features_1)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        augmented_images_1_s = tf.gather(augmented_images_1, shuffled_indices)
        features_1_s = tf.gather(features_1, shuffled_indices)
        diffdist = tf.reduce_mean(tf.square(augmented_images_1_s - augmented_images_2), axis=(1, 2, 3))
        weight_1 = tf.exp(-diffdist / self.temperature)
        dist_s_1 = tf.reduce_mean(tf.square(features_1_s - features_2), axis=1)

        indices = tf.range(start=0, limit=tf.shape(features_2)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        augmented_images_2_s = tf.gather(augmented_images_2, shuffled_indices)
        features_2_s = tf.gather(features_2, shuffled_indices)
        diffdist = tf.reduce_mean(tf.square(augmented_images_2_s - augmented_images_1), axis=(1, 2, 3))
        weight_2 = tf.exp(-diffdist / self.temperature)
        dist_s_2 = tf.reduce_mean(tf.square(features_2_s - features_1), axis=1)

        unsupervised_loss = tf.reduce_mean(weight_1 * dist_s_1) + tf.reduce_mean(weight_2 * dist_s_2)
        selfsupervised_loss = tf.reduce_mean(tf.square(features_1 - features_2))
        regularization_loss = tf.reduce_mean(tf.square(featureproduct - tf.eye(numfeature)))

        loss = self.tuneparameter[0] * unsupervised_loss / 2 + self.tuneparameter[1] * selfsupervised_loss + \
               self.tuneparameter[2] * regularization_loss

        return loss

    def train_step(self, data):
        (images, labels) = data
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
            "aiml_loss": aiml_loss*1000,
        }


def DownstreamTask(train_downstream_dataset, test_downstream_dataset, m_encoder, is_fine_tuning):
    epochs = 30
    backbone = tf.keras.Model(
        m_encoder.input, m_encoder.layers[-8].output
    )
    backbone.trainable = is_fine_tuning
    inputs = tf.keras.layers.Input(input_shape)
    x = backbone(inputs, training=is_fine_tuning)
    x = keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(1000, activation="softmax")(x)
    linear_model = tf.keras.Model(inputs, outputs, name="linear_model")

    linear_model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseTopKCategoricalAccuracy(k=5)],
        optimizer=tf.keras.optimizers.Adam(),
    )
    history = linear_model.fit(
        train_downstream_dataset, validation_data=test_downstream_dataset, epochs=epochs
    )
    _, test_acc = linear_model.evaluate(test_downstream_dataset)

    return test_acc


train_dataset, val_dataset = lin.load_data()

train_batch_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer).batch(batch_size, drop_remainder=True).prefetch(AUTO)

num_images = 10
images = next(iter(train_batch_dataset))[0][:num_images]
augmented_images = zip(
        images,
        augmenter(**aiml_augmenter)(images),
        n_augmenter(**null_augmenter)(images),
        classification_augmenter(**c_augmenter)(images),
    )
row_titles = [
        "Original:",
        "Strongly augmented:",
        "Strongly augmented:",
        "Weakly augmented:",
]
plt.figure(figsize=(num_images * 2.2, 4 * 2.2), dpi=100)
for column, image_row in enumerate(augmented_images):
    for row, image in enumerate(image_row):
        plt.subplot(4, num_images, row * num_images + column + 1)
        plt.imshow(image)
        if column == 0:
            plt.title(row_titles[row], loc="left")
        plt.axis("off")
plt.tight_layout()
plt.show()

model = AIML(temperature=1e-4, tuneparameter=(1, 1, 2e3), width=width)
model.compile(aiml_optimizer=keras.optimizers.Adam())
#model.encoder.summary()
pretrain_history = model.fit(train_batch_dataset, epochs=epochs)
model.encoder.save("AIML_Adam_500.keras")


# hyperparameters
num_epochs = 100
steps_per_epoch = 100
width = 2048

# hyperparameters corresponding to each algorithm
hyperparams = {
    methods.SimCLR: {"temperature": 0.1},
    methods.BarlowTwins: {"redundancy_reduction_weight": 10.0},
    methods.MoCo: {"momentum_coeff": 0.99, "temperature": 0.1, "queue_size": 10000},
    methods.NNCLR: {"temperature": 0.1, "queue_size": 10000},
}


# select an algorithm
Algorithm = methods.NNCLR

# architecture
model = Algorithm(
    contrastive_augmenter=keras.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            RandomResizedCrop(scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)),
            RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        ],
        name="contrastive_augmenter",
    ),
    encoder=resnet20.get_network(hidden_dim=width, use_pred=False, return_before_head=False),
    **hyperparams[Algorithm],
)


model.compile(contrastive_optimizer=keras.optimizers.Adam())
history = model.fit(train_batch_dataset, epochs=num_epochs)
model.encoder.save("NNCLR.keras")

ssl_methods = ["AIML", "SimCLR", "BarlowTwins", "MoCo", "NNCLR"]
samplesize_downstream = [1000, 4000, 16000, 64000]
steps_per_epoch = 100
results = []

for sample_num_train in samplesize_downstream:
    batch_size = sample_num_train // steps_per_epoch
    train_downstream_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer).take(sample_num_train).batch(batch_size, drop_remainder=True).prefetch(AUTO)
    test_downstream_dataset = val_dataset.shuffle(buffer_size=shuffle_buffer).batch(500, drop_remainder=True).prefetch(AUTO)
    for ssl_method in ssl_methods:
        algorithm_encoder = tf.keras.models.load_model(ssl_method + '.keras')
        tempresults1 = DownstreamTask(train_downstream_dataset, test_downstream_dataset, algorithm_encoder, False)
        tempresults2 = DownstreamTask(train_downstream_dataset, test_downstream_dataset, algorithm_encoder, True)
        results.append([tempresults1, tempresults2])




with open('comparison_imagenet_top5.txt', 'w') as f:
    for row in results:
        f.write("%s\n" % str(row))







