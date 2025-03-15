# 默认设置，在main.py中可以通过命令行参数传递覆盖
GLOBAL_CONFIG = {
    "height": 1060,
    "width": 1824,
    "step_size": 50,
    "end_lim": 50,
    "max_iter_time": 10,
    "max_search_time": 20,
    "path_len_diff": 1,
    "animation": False,
    "rewire": False,
    "image_path": "/data/ImageData/",
    "env_mode": "server"
}


def set_config(key: str, value):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG[key] = value
