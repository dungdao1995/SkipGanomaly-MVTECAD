from dataloader.dataloader import DataLoader
import tensorflow as tf
# internal
from .base_model import BaseModel
from .generator import Generator
from .discriminator import Discriminator
from executor.skipgan_trainer import SkipGanTrainer
from sklearn.metrics import f1_score, precision_score,roc_curve, auc
import numpy as np


class Skip_Ganomaly(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.generator = None
        self.discriminator = None

        #Training
        self.batch_size = self.config.train.batch_size
        self.epoches = self.config.train.epoches

        #Testing
        self.test_batch_size = self.config.test.batch_size

        #Data info
        self.image_size = self.config.data.image_size
        self.channels = self.config.data.channels

        #Dataset
        self.ds_train = None
        self.ds_test = None

        #save
        self.saved_path = 'saved_models/'

    def load_data(self):
        """Loads and Preprocess data """
        file_paths = DataLoader.file_paths(self.config)
        self.ds_train = DataLoader.preprocess_train(file_paths, self.batch_size,self.image_size,self.channels)
        self.ds_test = DataLoader.preprocess_test(self.config.data.test_path, self.image_size, self.test_batch_size)

    def build(self):
        self.generator = Generator()
        self.discriminator = Discriminator()

    def train(self):
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-3, 0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-3, 0.5)
        trainer = SkipGanTrainer(self.generator, self.discriminator, self.ds_train, self.generator_optimizer, self.discriminator_optimizer, self.epoches)
        trainer.train()

    def evaluate(self, set_lambda = 0.9):
        for x_test, y_test in self.ds_test:
            generated_images = self.generator(x_test, training=False)
            _, feat_real = self.discriminator(x_test, training=False)
            _, feat_fake = self.discriminator(generated_images, training=False)

            generated_images, feat_real, feat_fake = generated_images.numpy(), feat_real.numpy(), feat_fake.numpy()

            rec = np.abs(x_test-generated_images)
            # print(rec.shape)
            lat = np.square(feat_real-feat_fake)
            # print(lat.shape)


            rec = tf.reduce_sum(rec, [1,2,3])
            # print(rec.shape)
            lat = tf.reduce_sum(lat, [1,2,3])
            # print(lat.shape)

            error = set_lambda * rec + (1 - set_lambda) *lat
            error = (error - np.min(error)) / (np.max(error) - np.min(error)) # map to 0~1

            fpr, tpr, threshold = roc_curve(y_test, error)
            roc_auc = auc(fpr, tpr)

            return roc_auc,threshold

    def save_model(self):
        print(self.saved_path+'Generator')
        self.generator.save(self.saved_path+'Generator')
        self.discriminator.save(self.saved_path+'Discriminator')
        print('Model already saved!!!')


#==============Test Skip_Ganomaly Model============
from configs.config import CFG
if __name__ == '__main__':
    gan = Skip_Ganomaly(CFG)
    # gan.load_data()
    # gan.build()
    # for x, y in gan.ds_train.take(1):
    #     print(x.shape)
    gan.save_model()
    # gan.train()