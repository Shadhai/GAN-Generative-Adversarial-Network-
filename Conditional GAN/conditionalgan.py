import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')  # Use a backend that doesn't require a display
import matplotlib.pyplot as plt
# Generator model
def build_generator(latent_dim, num_classes):
    noise = layers.Input(shape=(latent_dim,))
    label = layers.Input(shape=(1,), dtype='int32')

    # Embedding label and flatten
    label_embedding = layers.Embedding(num_classes, latent_dim)(label)
    label_embedding = layers.Flatten()(label_embedding)

    # Multiply noise and label embedding element-wise
    model_input = layers.Multiply()([noise, label_embedding])

    x = layers.Dense(7 * 7 * 128, use_bias=False)(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((7, 7, 128))(x)
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    out = layers.Conv2D(1, kernel_size=7, activation='tanh', padding='same')(x)

    generator = tf.keras.Model([noise, label], out, name='generator')
    return generator

# Discriminator model
def build_discriminator(img_shape, num_classes):
    img = layers.Input(shape=img_shape)
    label = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(num_classes, np.prod(img_shape))(label)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape(img_shape)(label_embedding)

    concatenated = layers.Concatenate(axis=-1)([img, label_embedding])

    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(concatenated)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    discriminator = tf.keras.Model([img, label], out, name='discriminator')
    return discriminator

# Prepare dataset
def load_mnist():
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 127.5 - 1  # Normalize to [-1,1]
    x_train = np.expand_dims(x_train, axis=-1)       # (batch, 28, 28, 1)
    return x_train, y_train

# Save generated images during training
def save_images(generator, latent_dim, epoch, output_dir='cgan_output', num_classes=10, examples=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    noise = np.random.normal(0, 1, (examples, latent_dim))
    sampled_labels = np.arange(0, examples) % num_classes
    gen_imgs = generator.predict([noise, sampled_labels])

    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0,1]

    fig, axs = plt.subplots(1, examples, figsize=(examples * 2, 2))
    for i in range(examples):
        axs[i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(str(sampled_labels[i]))
    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f"{output_dir}/epoch_{epoch}.png")
    plt.close()

# Training function
def train(epochs, batch_size=64, latent_dim=100, num_classes=10):
    img_shape = (28, 28, 1)

    # Build models
    generator = build_generator(latent_dim, num_classes)
    discriminator = build_discriminator(img_shape, num_classes)

    # Compile discriminator
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                          metrics=['accuracy'])

    # Combined model (stacked generator and discriminator)
    noise = layers.Input(shape=(latent_dim,))
    label = layers.Input(shape=(1,), dtype='int32')
    generated_img = generator([noise, label])

    discriminator.trainable = False
    validity = discriminator([generated_img, label])

    combined = tf.keras.Model([noise, label], validity)
    combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

    # Load data
    X_train, y_train = load_mnist()
    half_batch = batch_size // 2

    for epoch in range(1, epochs + 1):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select random half batch of real images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs, labels = X_train[idx], y_train[idx]

        # Generate fake images
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        sampled_labels = np.random.randint(0, num_classes, half_batch)
        gen_imgs = generator.predict([noise, sampled_labels])

        # Labels for real and fake images
        real = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))

        # Train discriminator
        d_loss_real = discriminator.train_on_batch([imgs, labels], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, sampled_labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_labels = np.random.randint(0, num_classes, batch_size)
        valid_y = np.ones((batch_size, 1))  # try to fool discriminator

        g_loss = combined.train_on_batch([noise, sampled_labels], valid_y)

        # Print progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

        # Save images every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            save_images(generator, latent_dim, epoch, num_classes=num_classes)

if __name__ == "__main__":
    train(epochs=10, batch_size=64)
