import yaml
import os

config = None
with open('rrt_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

path_prefix = os.getenv('PATH_PREFIX', '/data/ImageData/')
