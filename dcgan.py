# %% --------------------------------------- Load Packages -------------------------------------------------------------
import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, \
    LeakyReLU, Conv2D, Conv2DTranspose, Embedding, \
    Concatenate, multiply, Flatten, BatchNormalization
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import Adam


# %% --------------------------------------- Fix Seeds -----------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_normal(seed=SEED)

# %% ---------------------------------- Data Preparation ---------------------------------------------------------------
def change_image_shape(images):
    shape_tuple = images.shape
    if len(shape_tuple) == 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], 1)
    elif shape_tuple == 4 and shape_tuple[-1] > 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], shape_tuple[1])
    return images
######################## MNIST ##########################
# # Load MNIST Fashion
from tensorflow.keras.datasets.fashion_mnist import load_data

# # Load training set
(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = change_image_shape(x_train), change_image_shape(x_test)
y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)

######################## Preprocessing ##########################
# Set channel
channel = x_train.shape[-1]

# It is suggested to use [-1, 1] input for GAN training
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5

# Get image size
img_size = x_train[0].shape

# Get number of classes
n_classes = len(np.unique(y_train))

# %% ---------------------------------- Hyperparameters ----------------------------------------------------------------

optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
latent_dim = 32
# trainRatio === times(Train D) / times(Train G)
# trainRatio = 5

# %% ---------------------------------- Models Setup -------------------------------------------------------------------
# Build Generator with mlp
def generator_fc():
    noise = Input(shape=(latent_dim,))
    x = Dense(128, kernel_initializer=weight_init)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256, kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(512, kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(np.prod(img_size), activation='tanh', kernel_initializer=weight_init)(x)
    out = Reshape(img_size)(x)
    model = Model(inputs=noise, outputs=out)
    return model

# Build Discriminator with mlp
def discriminator_fc():
    img = Input(shape=img_size)
    x = Flatten()(img)
    x = Dense(512, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(128, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    out = Dense(1, activation='sigmoid', kernel_initializer=weight_init)(x)

    model = Model(inputs=img, outputs=out)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build Generator with convolution layer
def generator_conv():
    noise = Input(shape=(latent_dim,))
    x = Dense(3 * 3 * 128)(noise)
    x = LeakyReLU(alpha=0.2)(x)

    ## Out size: 3 x 3 x 128
    x = Reshape((3, 3, 128))(x)

    ## Size: 7 x 7 x 128
    # remove padding='same' to scale 6x6 up to 7x7
    x = Conv2DTranspose(filters=128,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        # padding='same',
                        kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    ## Size: 14 x 14 x 64
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    ## Size: 28 x 28 x channel
    out = Conv2DTranspose(channel, (3, 3), activation='tanh', strides=(2, 2), padding='same',
                          kernel_initializer=weight_init)(x)

    model = Model(inputs=noise, outputs=out)
    return model

# Build Discriminator with convolution layer
def discriminator_conv():
    # 28 x 28 x channel
    img = Input(img_size)

    # 14 x 14 x 32
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(img)
    x = LeakyReLU(0.2)(x)

    # 7 x 7 x 64
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    # 3 x 3 x 128
    x = Conv2D(128, (3, 3), strides=(2, 2), kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=img, outputs=out)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def generator_trainer(generator, discriminator):
    # Freeze the discriminator when training generator
    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    return model


# %% ----------------------------------- GAN Part ----------------------------------------------------------------------
# Build our GAN
class GAN:
    def __init__(self, g_model, d_model):
        self.img_size = img_size  # channel_last
        self.z = latent_dim
        self.optimizer = optimizer

        self.generator = g_model
        self.discriminator = d_model

        self.train_gen = generator_trainer(self.generator, self.discriminator)
        self.loss_D, self.loss_G = [], []

    def train(self, imgs, steps_per_epoch=50, batch_size=128):
        # load data
        bs_half = batch_size//2

        for epoch in range(steps_per_epoch):
            # Get a half batch of random real images
            idx = np.random.randint(0, imgs.shape[0], bs_half)
            real_img = imgs[idx]

            # Generate a half batch of new images
            noise = np.random.normal(0, 1, size=(bs_half, latent_dim))
            fake_img = self.generator.predict(noise)

            # Train the discriminator
            loss_fake = self.discriminator.train_on_batch(fake_img, np.zeros(bs_half))
            loss_real = self.discriminator.train_on_batch(real_img, np.ones(bs_half))
            self.loss_D.append(0.5 * np.add(loss_fake, loss_real))

            # Train the generator
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            loss_gen = self.train_gen.train_on_batch(noise, np.ones(batch_size))
            self.loss_G.append(loss_gen)

            if (epoch + 1) * 10 % steps_per_epoch == 0:
                print('Steps (%d / %d): [Loss_D_real: %f, Loss_D_fake: %f, acc: %.2f%%] [Loss_G: %f]' %
                  (epoch+1, steps_per_epoch, loss_real[0], loss_fake[0], 100*self.loss_D[-1][1], loss_gen))

        return


# %% ----------------------------------- Compile Models ----------------------------------------------------------------
d_model = discriminator_conv()
g_model = generator_conv()

dcgan = GAN(g_model=g_model, d_model=d_model)

# %% ----------------------------------- Start Training ----------------------------------------------------------------
# Plot/save generated images through training
def plt_img(generator):
    np.random.seed(42)
    n = n_classes

    noise = np.random.normal(size=(n * n, latent_dim))
    decoded_imgs = generator.predict(noise)

    decoded_imgs = decoded_imgs * 0.5 + 0.5
    x_real = x_test * 0.5 + 0.5

    plt.figure(figsize=(2 * n, 2 * (n + 1)))
    for i in range(n):
        # display original
        ax = plt.subplot(n + 1, n, i + 1)
        if channel == 3:
            plt.imshow(x_real[y_test == i][0].reshape(img_size[0], img_size[1], img_size[2]))
        else:
            plt.imshow(x_real[y_test == i][0].reshape(img_size[0], img_size[1]))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for j in range(n):
            # display generation
            ax = plt.subplot(n + 1, n, (i + 1) * n + j + 1)
            if channel == 3:
                plt.imshow(decoded_imgs[i * n + j].reshape(img_size[0], img_size[1], img_size[2]))
            else:
                plt.imshow(decoded_imgs[i * n + j].reshape(img_size[0], img_size[1]))
                plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
    return

############################# Start training #############################
EPOCHS = 10
for epoch in range(EPOCHS):
    print('EPOCH # ', epoch + 1, '-' * 50)
    dcgan.train(x_train, steps_per_epoch=500, batch_size=128)
    if (epoch+1)%1 == 0:
        plt_img(dcgan.generator)

############################# Display performance #############################
# plot loss of G and D
plt.plot(np.array(dcgan.loss_D).T[0], label='D')
plt.plot(np.array(dcgan.loss_G), label='G')
plt.legend()
plt.show()