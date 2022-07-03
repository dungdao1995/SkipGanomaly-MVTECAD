import os

import tensorflow as tf
import time
from tqdm import tqdm
from .losses import generator_loss,discriminator_loss

class SkipGanTrainer:

    def __init__(self,generator, discriminator, ds_train, generator_optimizer, discriminator_optimizer, epoches):
        self.generator = generator
        self.discriminator = discriminator

        self.ds_train = ds_train

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.epoches = epoches
        self.model_save_path = '../app/model/'

    def train_step(self, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(images, training=True)

            pred_real, feat_real = self.discriminator(images, training=True)
            pred_fake, feat_fake = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(pred_real, pred_fake,
                                      images, generated_images,
                                      feat_real, feat_fake)

            disc_loss = discriminator_loss(pred_real, pred_fake)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self):
        steps = 0
        for epoch in range(self.epoches):
            start = time.time()

            for images, labels in tqdm(self.ds_train):
                steps += 1
                gen_loss, disc_loss = self.train_step(images)

                if steps % 100 == 0:
                    print ('Steps : {}, \t Total Gen Loss : {}, \t Total Dis Loss : {}'.format(steps, gen_loss.numpy(), disc_loss.numpy()))

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
