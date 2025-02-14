import yaml

config = None
with open('rrt_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
