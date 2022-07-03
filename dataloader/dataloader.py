import jsonschema
import tensorflow as tf
import pathlib
from configs.data_schema import SCHEMA

class DataLoader:
    """Data Loader class"""
    #=============Training DATA================
    @staticmethod
    def file_paths(config):
        """Loads dataset from path"""
        path = config.data.train_path
        file_paths = tf.data.Dataset.list_files(str(pathlib.Path(path + "*.png")))
        return file_paths

    @staticmethod
    def preprocess_train(file_paths, batch_size, image_size, channels):
        """ Preprocess and splits into training and test"""

        ds_train = file_paths.map(lambda file_path: DataLoader.process_path(file_path, image_size,channels),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
        return ds_train

    @staticmethod
    def process_path(file_path, image_size, channnels):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_png(image, channels=channnels)
        image = tf.image.resize(image, ((image_size, image_size)))
        #image = image/ 127 - 1
        image = image/ 255
        label = 1
        return image, label

    #=============Test DATA================
    @staticmethod
    def preprocess_test(path, image_size, batch_size):
        ds_test = tf.keras.preprocessing.image_dataset_from_directory(
            path,
            labels = 'inferred',
            label_mode = 'int', #categorical, binary
            #class_names = ['1', '0'...]
            color_mode = 'grayscale',
            batch_size = batch_size,
            image_size = (image_size, image_size), #reshape if not in this size
            shuffle = True,
            seed = 123,
        )

        ds_test = ds_test.map(DataLoader.process_test)
        return ds_test

    @staticmethod
    def process_test(image, label):
        image = tf.cast(image/255., tf.float32)
        return image,label

    #=================Check the DATA-Schema===========
    @staticmethod
    def validate_schema(data_point):
        jsonschema.validate({'image': data_point.tolist()}, SCHEMA)


#==========Testing============
from configs.config import CFG
from utils.config import Config

if __name__ == '__main__':
    config = Config.from_json(CFG)
    file_paths = DataLoader.file_paths(config)

    ds_train = DataLoader.preprocess_train(file_paths,
                                           config.train.batch_size,
                                           config.data.image_size,
                                           config.data.channels)
    # for x,y in ds_train.take(1):
    #     print(x.shape)
    #     print(y)
    ds_test = DataLoader.preprocess_test(config.data.test_path,
                                         config.data.image_size,
                                         config.test.batch_size)
    # for x,y in ds_test.take(1):
    #     print(x.shape)
    #     print(y)

