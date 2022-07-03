# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "train_path": "data/screw/train/good/",
        "test_path": "data/screw/test/",
        "image_size": 128,
        "channels" : 1,
    },
    "train": {
        "batch_size": 4,
        "buffer_size": 1000,
        "epoches": 10,
        "optimizer": {
            "type": "adam",
            "learning_rate": 2e-3,
            "beta_1": 0.5,
        },
    },
    "test": {
        "batch_size": 160,
    },
    "model": {
        "input": [128, 128, 1],
        "g_encoder": {
            "layer_1": 64,
            "layer_2": 128,
            "layer_3": 256,
            "layer_4": 512,
            'center': 512,
        },
        "g_decoder": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            'output': 1,
        },
        "discriminator": {
            "layer_1": 64,
            "layer_2": 128,
            "layer_3": 256,
            "layer_4": 512,
            'center': 100,
            'output': 1,
        },
    }
}