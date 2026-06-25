"""Shared Keras model definition for collaborative filtering."""

import tensorflow as tf


class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.movie_embedding = tf.keras.layers.Embedding(
            num_movies, embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.movie_bias = tf.keras.layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vec = self.user_embedding(inputs[:, 0])
        user_b = self.user_bias(inputs[:, 0])
        movie_vec = self.movie_embedding(inputs[:, 1])
        movie_b = self.movie_bias(inputs[:, 1])
        dot = tf.reduce_sum(user_vec * movie_vec, axis=1, keepdims=True)
        return dot + user_b + movie_b

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_movies": self.num_movies,
            "embedding_size": self.embedding_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
