##The implementation here is modified version of code in https://github.com/beresandras/contrastive-classification-keras/tree/master.

import tensorflow as tf
import random
import tensorflow_datasets as tfds


from abc import abstractmethod
from tensorflow import keras
from tensorflow.keras import layers

class ContrastiveModel(keras.Model):
    def __init__(
        self,
        contrastive_augmenter,
        #classification_augmenter,
        encoder,
        #linear_probe,
    ):
        super().__init__()

        self.contrastive_augmenter = contrastive_augmenter
        #self.classification_augmenter = classification_augmenter
        self.encoder = encoder
        #self.linear_probe = linear_probe

    def compile(self, contrastive_optimizer, **kwargs):
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
        (images, labels) = data
        # each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            # the representations are passed through a projection mlp
            projections_1 = self.encoder(augmented_images_1)
            projections_2 = self.encoder(augmented_images_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights,
            )
        )

        return {
            "c_loss": contrastive_loss,
        }



class SimCLR(ContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        #classification_augmenter,
        encoder,
        #linear_probe,
        temperature,
    ):
        super().__init__(
            contrastive_augmenter,
            #classification_augmenter,
            encoder,
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
        #linear_probe,
        redundancy_reduction_weight,
    ):
        super().__init__(
            contrastive_augmenter,
            #classification_augmenter,
            encoder,
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
        #linear_probe,
        momentum_coeff,
    ):
        super().__init__(
            contrastive_augmenter,
            #classification_augmenter,
            encoder,
            #linear_probe,
        )
        self.momentum_coeff = momentum_coeff

        # the momentum networks are initialized from their online counterparts
        self.m_encoder = keras.models.clone_model(self.encoder)

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
        (images, labels) = data
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            projections_1 = self.encoder(augmented_images_1)
            projections_2 = self.encoder(augmented_images_2)
            m_projections_1 = self.m_encoder(augmented_images_1)
            m_projections_2 = self.m_encoder(augmented_images_2)
            contrastive_loss = self.contrastive_loss(
                projections_1, projections_2, m_projections_1, m_projections_2
            )
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights,
            )
        )

        # the momentum networks are updated by exponential moving average
        for weight, m_weight in zip(self.encoder.weights, self.m_encoder.weights):
            m_weight.assign(
                self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight
            )


        return {
            "c_loss": contrastive_loss,
        }



class MoCo(MomentumContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        #classification_augmenter,
        encoder,
        #linear_probe,
        momentum_coeff,
        temperature,
        queue_size,
    ):
        super().__init__(
            contrastive_augmenter,
            #classification_augmenter,
            encoder,
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
        #linear_probe,
        temperature,
        queue_size,
    ):
        super().__init__(
            contrastive_augmenter,
            #classification_augmenter,
            encoder,
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






