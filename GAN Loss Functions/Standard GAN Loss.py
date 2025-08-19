import tensorflow as tf
from tensorflow.keras import layers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss
# Type	Generator Loss	Discriminator Loss
# Standard	BCE(1, D(G(z)))	BCE(1, D(x)) + BCE(0, D(G(z)))
if __name__ == "__main__":
    import tensorflow as tf

    # Simulated outputs from a discriminator
    fake_output = tf.random.uniform([4, 1], minval=0, maxval=1)
    real_output = tf.random.uniform([4, 1], minval=0.8, maxval=1.0)

    g_loss = generator_loss(fake_output)
    d_loss = discriminator_loss(real_output, fake_output)

    print("Generator Loss:", g_loss.numpy())
    print("Discriminator Loss:", d_loss.numpy())
