
import yaml
import os

def load_config_file(filename):
    filepath = os.path.join('./experiments', filename+'.yml')

    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

        # yaml parses scientific notation as str, not float
        cfg['learning_rate'] = float(cfg['learning_rate'])
        cfg['weight_decay'] = float(cfg['weight_decay'])

        return cfg
