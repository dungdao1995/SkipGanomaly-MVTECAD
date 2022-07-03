# -*- coding: utf-8 -*-
"""Config class"""

import json
from configs.config import CFG

class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self, data, train, test,model):
        self.data = data
        self.train = train
        self.test = test
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.test, params.model)


class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)

#Testing the function
if __name__ == '__main__':
    config = Config.from_json(CFG)
    print("Image_size: ",config.data.image_size)
    print("Channels: ",config.data.channels)
    print("Channels: ",config.test.batch_size)