import os
import yaml

CUR_DIR = os.path.dirname(__file__)

def get_config():
    with open('./app.yml', encoding='utf-8') as cfgFile:
        config_app = yaml.safe_load(cfgFile)        
    return config_app