import yaml


class LoadConfig():
    def __init__(self, config_path):
        with open(config_path) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

    def train_config(self):
        train_cfg = self.config['TRAIN_CONFIG']
        image_config, model_config, run_config = train_cfg['IMAGE_CONFIG'], train_cfg['MODEL_CONFIG'], train_cfg['RUN_CONFIG']
        return image_config, model_config, run_config

    def test_config(self):
        test_cfg = self.config['TEST_CONFIG']
        return test_cfg
