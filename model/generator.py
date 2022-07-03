import tensorflow as tf
from tensorflow.keras import layers
from .layer import Conv_block,Conv_T_block
from configs.config import CFG
from utils.config import Config



config = Config.from_json(CFG)

g_encoder_layer1 = config.model.g_encoder.layer_1
g_encoder_layer2 = config.model.g_encoder.layer_2
g_encoder_layer3 = config.model.g_encoder.layer_3
g_encoder_layer4 = config.model.g_encoder.layer_4
g_encoder_center = config.model.g_encoder.center
# print(g_encoder_layer1)
# print(g_encoder_layer2)
# print(g_encoder_layer3)
# print(g_encoder_layer4)
# print(g_encoder_center)

g_decoder_layer1 = config.model.g_decoder.layer_1
g_decoder_layer2 = config.model.g_decoder.layer_2
g_decoder_layer3 = config.model.g_decoder.layer_3
g_decoder_layer4 = config.model.g_decoder.layer_4
g_decoder_output = config.model.g_decoder.output
# print(g_decoder_layer1)
# print(g_decoder_layer2)
# print(g_decoder_layer3)
# print(g_decoder_layer4)
# print(g_decoder_output)

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder_1 = Conv_block(g_encoder_layer1) # 16
        self.encoder_2 = Conv_block(g_encoder_layer2) # 8
        self.encoder_3 = Conv_block(g_encoder_layer3) # 4
        self.encoder_4 = Conv_block(g_encoder_layer4) # 2

        self.center = Conv_block(g_encoder_center) # 1

        self.decoder_4 = Conv_T_block(g_decoder_layer1) # 2
        self.decoder_3 = Conv_T_block(g_decoder_layer2) # 4
        self.decoder_2 = Conv_T_block(g_decoder_layer3) # 8
        self.decoder_1 = Conv_T_block(g_decoder_layer4) # 16

        self.output_layer = layers.Conv2DTranspose(g_decoder_output, 1, strides=2, padding='same', use_bias=False, # 32
                                                   kernel_initializer=tf.random_normal_initializer(0., 0.02))

    def call(self, inputs, training=False):
        en_1 = self.encoder_1(inputs) # gen
        en_2 = self.encoder_2(en_1)
        en_3 = self.encoder_3(en_2)
        en_4 = self.encoder_4(en_3)

        center = self.center(en_4)

        de_4 = self.decoder_4(center, en_4)
        de_3 = self.decoder_3(de_4, en_3)
        de_2 = self.decoder_2(de_3, en_2)
        de_1 = self.decoder_1(de_2, en_1)

        outputs = self.output_layer(de_1)

        return outputs