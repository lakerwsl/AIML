import tensorflow as tf
import random
import tensorflow_datasets as tfds

from abc import abstractmethod
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

class ContrastiveModel(keras.Model):
    def __init__(
        self,
        contrastive_augmenter,
        #classification_augmenter,
        encoder,
        projection_head,
        #linear_probe,
    ):
        super().__init__()

        self.contrastive_augmenter = contrastive_augmenter
        #self.classification_augmenter = classification_augmenter
        self.encoder = encoder
        self.projection_head = projection_head
        #self.linear_probe = linear_probe

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        #self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        #self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        #self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy()
        #self.correlation_accuracy = keras.metrics.SparseCategoricalAccuracy()
        #self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy()


    @abstractmethod
    def contrastive_loss(self, projections_1, projections_2):
        pass

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data
        # both labeled and unlabeled images are used, without labels
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        # each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            # the representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        # self.update_contrastive_accuracy(features_1, features_2)
        # self.update_correlation_accuracy(features_1, features_2)

        # labels are only used in evalutation for an on-the-fly logistic regression
        # preprocessed_images = self.classification_augmenter(labeled_images)
        # with tf.GradientTape() as tape:
        #     features = self.encoder(preprocessed_images)
        #     class_logits = self.linear_probe(features)
        #     probe_loss = self.probe_loss(labels, class_logits)
        # gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        # self.probe_optimizer.apply_gradients(
        #     zip(gradients, self.linear_probe.trainable_weights)
        # )
        # self.probe_accuracy.update_state(labels, class_logits)

        return {
            "c_loss": contrastive_loss,
            #"c_acc": self.contrastive_accuracy.result(),
            #"r_acc": self.correlation_accuracy.result(),
            #"p_loss": probe_loss,
            #"p_acc": self.probe_accuracy.result(),
        }



class SimCLR(ContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        #classification_augmenter,
        encoder,
        projection_head,
        #linear_probe,
        temperature,
    ):
        super().__init__(
            contrastive_augmenter,
            #classification_augmenter,
            encoder,
            projection_head,
            #linear_probe,
        )
        self.temperature = temperature

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # the temperature-scaled similarities are used as logits for cross-entropy
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
            from_logits=True,
        )
        return loss


class BarlowTwins(ContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        #classification_augmenter,
        encoder,
        projection_head,
        #linear_probe,
        redundancy_reduction_weight,
    ):
        super().__init__(
            contrastive_augmenter,
            #classification_augmenter,
            encoder,
            projection_head,
            #linear_probe,
        )
        # weighting coefficient between the two loss components
        self.redundancy_reduction_weight = redundancy_reduction_weight
        # its value differs from the paper, because the loss implementation has been
        # changed to be invariant to the encoder output dimensions (feature dim)

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = (
            projections_1 - tf.reduce_mean(projections_1, axis=0)
        ) / tf.math.reduce_std(projections_1, axis=0)
        projections_2 = (
            projections_2 - tf.reduce_mean(projections_2, axis=0)
        ) / tf.math.reduce_std(projections_2, axis=0)

        # the cross correlation of image representations should be the identity matrix
        batch_size = tf.shape(projections_1, out_type=tf.float32)[0]
        feature_dim = tf.shape(projections_1, out_type=tf.float32)[1]
        cross_correlation = (
            tf.matmul(projections_1, projections_2, transpose_a=True) / batch_size
        )
        target_cross_correlation = tf.eye(feature_dim)
        squared_errors = (target_cross_correlation - cross_correlation) ** 2

        # invariance loss = average diagonal error
        # redundancy reduction loss = average off-diagonal error
        invariance_loss = (
            tf.reduce_sum(squared_errors * tf.eye(feature_dim)) / feature_dim
        )
        redundancy_reduction_loss = tf.reduce_sum(
            squared_errors * (1 - tf.eye(feature_dim))
        ) / (feature_dim * (feature_dim - 1))
        return (
            invariance_loss
            + self.redundancy_reduction_weight * redundancy_reduction_loss
        )















class MomentumContrastiveModel(ContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        #classification_augmenter,
        encoder,
        projection_head,
        #linear_probe,
        momentum_coeff,
    ):
        super().__init__(
            contrastive_augmenter,
            #classification_augmenter,
            encoder,
            projection_head,
            #linear_probe,
        )
        self.momentum_coeff = momentum_coeff

        # the momentum networks are initialized from their online counterparts
        self.m_encoder = keras.models.clone_model(self.encoder)
        self.m_projection_head = keras.models.clone_model(self.projection_head)

    @abstractmethod
    def contrastive_loss(
        self,
        projections_1,
        projections_2,
        m_projections_1,
        m_projections_2,
    ):
        pass

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            m_features_1 = self.m_encoder(augmented_images_1)
            m_features_2 = self.m_encoder(augmented_images_2)
            m_projections_1 = self.m_projection_head(m_features_1)
            m_projections_2 = self.m_projection_head(m_features_2)
            contrastive_loss = self.contrastive_loss(
                projections_1, projections_2, m_projections_1, m_projections_2
            )
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        # self.update_contrastive_accuracy(m_features_1, m_features_2)
        # self.update_correlation_accuracy(m_features_1, m_features_2)
        # 
        # preprocessed_images = self.classification_augmenter(labeled_images)
        # with tf.GradientTape() as tape:
        #     # the momentum encoder is used here as it moves more slowly
        #     features = self.m_encoder(preprocessed_images)
        #     class_logits = self.linear_probe(features)
        #     probe_loss = self.probe_loss(labels, class_logits)
        # gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        # self.probe_optimizer.apply_gradients(
        #     zip(gradients, self.linear_probe.trainable_weights)
        # )
        # self.probe_accuracy.update_state(labels, class_logits)

        # the momentum networks are updated by exponential moving average
        for weight, m_weight in zip(self.encoder.weights, self.m_encoder.weights):
            m_weight.assign(
                self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight
            )
        for weight, m_weight in zip(
            self.projection_head.weights, self.m_projection_head.weights
        ):
            m_weight.assign(
                self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight
            )

        return {
            "c_loss": contrastive_loss,
            # "c_acc": self.contrastive_accuracy.result(),
            # "r_acc": self.correlation_accuracy.result(),
            # "p_loss": probe_loss,
            # "p_acc": self.probe_accuracy.result(),
        }



class MoCo(MomentumContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        #classification_augmenter,
        encoder,
        projection_head,
        #linear_probe,
        momentum_coeff,
        temperature,
        queue_size,
    ):
        super().__init__(
            contrastive_augmenter,
            #classification_augmenter,
            encoder,
            projection_head,
            #linear_probe,
            momentum_coeff,
        )
        self.temperature = temperature

        feature_dimensions = encoder.output_shape[1]
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1
            ),
            trainable=False,
        )

    def contrastive_loss(
        self,
        projections_1,
        projections_2,
        m_projections_1,
        m_projections_2,
    ):
        # similar to the SimCLR loss, however it uses the momentum networks'
        # representations of the differently augmented views as targets
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        m_projections_1 = tf.math.l2_normalize(m_projections_1, axis=1)
        m_projections_2 = tf.math.l2_normalize(m_projections_2, axis=1)

        similarities_1_2 = (
            tf.matmul(
                projections_1,
                tf.concat((m_projections_2, self.feature_queue), axis=0),
                transpose_b=True,
            )
            / self.temperature
        )
        similarities_2_1 = (
            tf.matmul(
                projections_2,
                tf.concat((m_projections_1, self.feature_queue), axis=0),
                transpose_b=True,
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities_1_2, similarities_2_1], axis=0),
            from_logits=True,
        )

        # feature queue update
        self.feature_queue.assign(
            tf.concat(
                [
                    m_projections_1,
                    m_projections_2,
                    self.feature_queue[: -(2 * batch_size)],
                ],
                axis=0,
            )
        )
        return loss


class NNCLR(ContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        #classification_augmenter,
        encoder,
        projection_head,
        #linear_probe,
        temperature,
        queue_size,
    ):
        super().__init__(
            contrastive_augmenter,
            #classification_augmenter,
            encoder,
            projection_head,
            #linear_probe,
        )
        self.temperature = temperature

        feature_dimensions = encoder.output_shape[1]
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1
            ),
            trainable=False,
        )

    def nearest_neighbour(self, projections):
        # highest cosine similarity == lowest L2 distance, for L2 normalized features
        support_similarities = tf.matmul(
            projections, self.feature_queue, transpose_b=True
        )

        # hard nearest-neighbours
        nn_projections = tf.gather(
            self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0
        )

        # straight-through gradient estimation
        # paper used stop gradient, however it helps performance at this scale
        return projections + tf.stop_gradient(nn_projections - projections)

    def contrastive_loss(self, projections_1, projections_2):
        # similar to the SimCLR loss, however we take the nearest neighbours of a set
        # of projections from a feature queue
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities_1_2 = (
            tf.matmul(
                self.nearest_neighbour(projections_1), projections_2, transpose_b=True
            )
            / self.temperature
        )
        similarities_2_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_2), projections_1, transpose_b=True
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities_1_2, similarities_2_1], axis=0),
            from_logits=True,
        )

        # feature queue update
        self.feature_queue.assign(
            tf.concat([projections_1, self.feature_queue[:-batch_size]], axis=0)
        )
        return loss



# hyperparameters
num_epochs = 30
steps_per_epoch = 200
width = 128

# hyperparameters corresponding to each algorithm
hyperparams = {
    SimCLR: {"temperature": 0.1},
    BarlowTwins: {"redundancy_reduction_weight": 10.0},
    MoCo: {"momentum_coeff": 0.99, "temperature": 0.1, "queue_size": 10000},
    NNCLR: {"temperature": 0.1, "queue_size": 10000},
}


# select an algorithm
Algorithm = NNCLR

# architecture
model = Algorithm(
    contrastive_augmenter=keras.Sequential(
        [
            layers.Input(shape=(96, 96, 3)),
            preprocessing.Rescaling(1 / 255),
            preprocessing.RandomFlip("horizontal"),
            RandomResizedCrop(scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)),
            RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        ],
        name="contrastive_augmenter",
    ),
    classification_augmenter=keras.Sequential(
        [
            layers.Input(shape=(96, 96, 3)),
            preprocessing.Rescaling(1 / 255),
            preprocessing.RandomFlip("horizontal"),
            RandomResizedCrop(scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
            RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ],
        name="classification_augmenter",
    ),
    encoder=keras.Sequential(
        [
            layers.Input(shape=(96, 96, 3)),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(width, activation="relu"),
        ],
        name="encoder",
    ),
    projection_head=keras.Sequential(
        [
            layers.Input(shape=(width,)),
            layers.Dense(width, activation="relu"),
            layers.Dense(width),
        ],
        name="projection_head",
    ),
    linear_probe=keras.Sequential(
        [
            layers.Input(shape=(width,)),
            layers.Dense(10),
        ],
        name="linear_probe",
    ),
    **hyperparams[Algorithm],
)

# optimizers
model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
)

# run training
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

# save history
with open("{}.pkl".format(Algorithm.__name__), "wb") as write_file:
    pickle.dump(history.history, write_file)
