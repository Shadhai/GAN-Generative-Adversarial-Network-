import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a backend that doesn't require a display
import matplotlib.pyplot as plt

# Load CIFAR-10 data
(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train[:1000]  # Use a small subset for speed

# Define a simple self-attention layer
class SelfAttention(layers.Layer):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query_conv = layers.Conv2D(channels // 8, 1)
        self.key_conv = layers.Conv2D(channels // 8, 1)
        self.value_conv = layers.Conv2D(channels, 1)
        self.gamma = self.add_weight(name="gamma", shape=[1], initializer="zeros", trainable=True)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]

        proj_query = tf.reshape(self.query_conv(x), [batch_size, -1, height * width])
        proj_key = tf.reshape(self.key_conv(x), [batch_size, -1, height * width])
        energy = tf.matmul(tf.transpose(proj_query, [0, 2, 1]), proj_key)
        attention = tf.nn.softmax(energy, axis=-1)
        proj_value = tf.reshape(self.value_conv(x), [batch_size, -1, height * width])
        out = tf.matmul(proj_value, tf.transpose(attention, [0, 2, 1]))
        out = tf.reshape(out, shape=[batch_size, height, width, channels])
        return self.gamma * out + x


# Define the autoencoder with self-attention
def build_autoencoder():
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = SelfAttention(32)(x)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D()(x)

    x = layers.Conv2DTranspose(16, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D()(x)
    x = SelfAttention(16)(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D()(x)
    decoded = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    return models.Model(inputs, decoded)

autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

# Train and validate
history = autoencoder.fit(x_train, x_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

# Plot MSE loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train MSE')
plt.plot(history.history['val_loss'], label='Val MSE')
plt.title("Training vs Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("mse_plot.png")
print("Saved plot to mse_plot.png")

