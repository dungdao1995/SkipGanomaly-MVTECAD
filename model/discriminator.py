import tensorflow as tf
from tensorflow.keras import layers
from .layer import Conv_block,Conv_T_block
from configs.config import CFG
from utils.config import Config



config = Config.from_json(CFG)

discriminator_layer1 = config.model.discriminator.layer_1
discriminator_layer2 = config.model.discriminator.layer_2
discriminator_layer3 = config.model.discriminator.layer_3
discriminator_layer4 = config.model.discriminator.layer_4
discriminator_center = config.model.discriminator.center
discriminator_output = config.model.discriminator.output
# print(discriminator_layer1)
# print(discriminator_layer2)
# print(discriminator_layer3)
# print(discriminator_layer4)
# print(discriminator_center)
# print(discriminator_output)

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder_1 = Conv_block(discriminator_layer1) # 16
        self.encoder_2 = Conv_block(discriminator_layer2) # 8
        self.encoder_3 = Conv_block(discriminator_layer3) # 4
        self.encoder_4 = Conv_block(discriminator_layer4) # 2

        self.center = Conv_block(discriminator_center) # 1

        self.outputs = layers.Conv2D(discriminator_output, 3, strides=1, padding='same',
                                     use_bias=False, activation='sigmoid')

    def call(self, inputs, training=False):
        en_1 = self.encoder_1(inputs) # dis
        en_2 = self.encoder_2(en_1)
        en_3 = self.encoder_3(en_2)
        en_4 = self.encoder_4(en_3)

        center = self.center(en_4)

        outputs = self.outputs(center)

        return outputs, center

