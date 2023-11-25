from sklearn.neighbors import KNeighborsClassifier
import keras_cv

input_shape = (96, 96, 3)
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 96
dataset_name = "stl10"
null_augmenter = {"name": "null_augmenter",}
aiml_augmenter = {"name": "aiml_augmenter",}
c_augmenter = {"name": "classification_augmenter",}
shuffle_buffer = 5000
labelled_train_images = 5000
unlabelled_images = 100000
steps_per_epoch = 200
width = 128
unlabeled_batch_size = unlabelled_images // steps_per_epoch
labeled_batch_size = labelled_train_images // steps_per_epoch
batch_size = unlabeled_batch_size + labeled_batch_size

ds = tfds.load(dataset_name, split="train", as_supervised=True, batch_size=-1)
train_examples, train_labels = tfds.as_numpy(ds)
ds = tfds.load(dataset_name, split="test", as_supervised=True, batch_size=-1)
test_examples, test_labels = tfds.as_numpy(ds)


unlabeled_train_dataset = (
  tfds.load(dataset_name, split="unlabelled", as_supervised=True, shuffle_files=True)
  .shuffle(buffer_size=shuffle_buffer)
  .batch(unlabeled_batch_size, drop_remainder=True)
)
labeled_train_dataset = (
  tfds.load(dataset_name, split="train", as_supervised=True, shuffle_files=True)
  .shuffle(buffer_size=shuffle_buffer)
  .batch(labeled_batch_size, drop_remainder=True)
)
test_dataset = (
  tfds.load(dataset_name, split="test", as_supervised=True)
  .batch(batch_size)
  .prefetch(buffer_size=AUTOTUNE)
)
train_dataset = tf.data.Dataset.zip((unlabeled_train_dataset, labeled_train_dataset)).prefetch(buffer_size=AUTOTUNE)


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

def encoder(width):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
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
        (unlabeled_images, _), (labeled_images, labels) = data
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
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

def DownstreamTask(m_encoder, train_examples, train_labels, test_examples, test_labels, steps_per_epoch):
  samplesize_downstream = [1000, 3000, 5000]
  resulting_encoder = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        classification_augmenter(**c_augmenter),
        m_encoder,
    ],
    name="resulting_model",
  )
  x_train_representation = resulting_encoder(train_examples)
  x_test_representation = resulting_encoder(test_examples)
  idxs = tf.range(tf.shape(train_examples)[0])
  idxs_t = tf.range(tf.shape(test_examples)[0])
  results = []
  
  for i in range(0,len(samplesize_downstream)):
    sample_num_train = samplesize_downstream[i]
    down_batch_size = sample_num_train // steps_per_epoch
    ridxs = tf.random.shuffle(idxs)[:sample_num_train]
    ridxs_t = tf.random.shuffle(idxs_t)[:sample_num_train]
    train_dataset2 = tf.data.Dataset.from_tensor_slices((tf.gather(train_examples, ridxs), tf.gather(train_labels, ridxs)))
    train_dataset2 = train_dataset2.shuffle(buffer_size=shuffle_buffer).batch(down_batch_size, drop_remainder=True)
    test_dataset2 = tf.data.Dataset.from_tensor_slices((tf.gather(test_examples, ridxs_t), tf.gather(test_labels, ridxs_t)))
    test_dataset2 = test_dataset2.shuffle(buffer_size=shuffle_buffer).batch(down_batch_size, drop_remainder=True)
    finetuning_model = keras.Sequential(
      [
        layers.Input(shape=input_shape),
        classification_augmenter(**c_augmenter),
        m_encoder,
        layers.Dense(10),
      ],
      name="finetuning_model",
    )
    finetuning_model.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    finetuning_history = finetuning_model.fit(
      train_dataset2, epochs=30, validation_data=test_dataset2
    )
    accuracy1 = finetuning_model.evaluate(test_dataset2)
    
    finetuning_model = keras.Sequential(
      [
        layers.Input(shape=input_shape),
        classification_augmenter(**c_augmenter),
        m_encoder,
        layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.02)),
      ],
      name="finetuning_model",
    )
    finetuning_model.layers[0].trainable = False
    finetuning_model.layers[1].trainable = False
    finetuning_model.compile(
      optimizer=tfa.optimizers.LAMB(),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    finetuning_history = finetuning_model.fit(
      train_dataset2, epochs=30, validation_data=test_dataset2
    )
    accuracy2 = finetuning_model.evaluate(test_dataset2)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(tf.gather(x_train_representation, ridxs), tf.gather(train_labels, ridxs))
    accuracy3 = knn.score(tf.gather(x_test_representation, ridxs_t), tf.gather(test_labels, ridxs_t))
    results.append([sample_num_train, accuracy1[1], accuracy2[1], accuracy3])
    
  return results

  
  


model = AIML(temperature=0.0001, tuneparameter = (1,1,200), width=width)
model.compile(aiml_optimizer=keras.optimizers.Adam())
pretrain_history = model.fit(train_dataset, epochs=100)

results2=DownstreamTask(model.encoder, train_examples, train_labels, test_examples, test_labels, 100)
np.savez('comparison_stl10_aiml.npz', results=results2)

# hyperparameters
num_epochs = 100
steps_per_epoch = 200
width = 128

# hyperparameters corresponding to each algorithm
hyperparams = {
    SimCLR: {"temperature": 0.1},
    BarlowTwins: {"redundancy_reduction_weight": 0.1},
    MoCo: {"momentum_coeff": 0.99, "temperature": 0.1, "queue_size": 10000},
    NNCLR: {"temperature": 0.1, "queue_size": 10000},
}

Algorithms = [SimCLR, BarlowTwins, MoCo, NNCLR]
Algorithm = SimCLR
model = Algorithm(
    augmenter(**aiml_augmenter),
    encoder=encoder(width),
    projection_head=keras.Sequential(
        [
            layers.Input(shape=(width,)),
            layers.Dense(width),
        ],
        name="projection_head",
    ),
    **hyperparams[Algorithm],
)
model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
)
pretrain_history = model.fit(train_dataset, epochs=num_epochs)
results2=DownstreamTask(model.encoder, train_examples, train_labels, test_examples, test_labels, 100)
np.savez('comparison_stl10_simclr.npz', results=results2)




