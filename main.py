# This is a sample Python script.

from configs.config import CFG
from model.skip_ganomaly import Skip_Ganomaly


def run():
    """Builds model, loads data, trains and evaluates"""
    model = Skip_Ganomaly(CFG)
    model.load_data()
    model.build()
    model.train()
    result,threshold = model.evaluate()
    model.save_model()
    print('Area under the Curve: ',result )
    print('Threshold: ',threshold )


if __name__ == '__main__':
    run()

