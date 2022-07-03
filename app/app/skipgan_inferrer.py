from PIL import Image
import tensorflow as tf
import keras
import numpy as np

class SkipganInferrer:
    def __init__(self):
        self.image_size = 128

        self.gen_path = '../../saved_models/Generator'
        self.disc_path = '../../saved_models/Discriminator'
        self.generator = keras.models.load_model(self.gen_path)
        self.discriminator = keras.models.load_model(self.disc_path)


    def preprocessing_data(self,file):
        image = Image.open(file)
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image)
        test = image.reshape(1, self.image_size, self.image_size, 1)
        test = test/255
        a = tf.convert_to_tensor(test,dtype=tf.float32)
        return a

    def infer(self, image):
        image = self.preprocessing_data(image)
        set_lambda = 0.9 #set lamda in anomaly score
        generated_images = self.generator(image, training=False)
        _, feat_real = self.discriminator(image, training=False)
        _, feat_fake = self.discriminator(generated_images, training=False)

        generated_images, feat_real, feat_fake = generated_images.numpy(), feat_real.numpy(), feat_fake.numpy()

        rec = np.abs(image-generated_images)
        # print(rec.shape)
        lat = np.square(feat_real-feat_fake)
        # print(lat.shape)


        rec = tf.reduce_sum(rec, [1,2,3])
        # print(rec.shape)
        lat = tf.reduce_sum(lat, [1,2,3])
        # print(lat.shape)

        error = set_lambda * rec + (1 - set_lambda) *lat
        # print(np.min(error))
        # print(np.max(error))
        error = (error - 8606.58) / (9985.801 - 8606.58) # map to 0~1
        print(error)
        # thre9187shold = 0.22424348
        threshold = 0.2672

        if np.array(error)[0] >= threshold:
            return 'Abnormal'
        else:
            return 'Normal'