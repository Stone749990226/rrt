from datetime import datetime, timedelta
import os
from PIL import Image
from fastapi import HTTPException
import math
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import logging
import numpy as np
import time
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from scipy.ndimage import binary_dilation
import re

from config import config
maps = {}
lookup_table = np.load('cloud_latlon_lookup_table_average.npy')


def align_time_15m(time_str: str):
    time_obj = datetime.strptime(time_str, "%Y%m%d%H%M")
    aligned_time = time_obj.replace(
        minute=(time_obj.minute // 15) * 15, second=0, microsecond=0)
    return aligned_time.strftime("%Y%m%d%H%M")


def get_images_path(start_time, mark_time, prefix="/data/ImageData/"):
    """/data/ImageData/20241206/11/cloud_dugs_unet_3h/16-45"""
    res = []
    start_time_obj = datetime.strptime(start_time, "%Y%m%d%H%M")
    if config["env_mode"] == "local":
        for root, dirs, files in os.walk("./07-00"):
            for file in files:
                if file.endswith('.png'):
                    time_str = file.split('.')[0]
                    time_obj = datetime.strptime(time_str, "%Y%m%d%H%M")
                    if time_obj > start_time_obj - timedelta(minutes=15):
                        res.append(os.path.abspath(os.path.join(root, file)))
        if len(res) == 0:
            raise HTTPException(
                status_code=404, detail=f"Image not found for the dir: {dir}")
        res.sort()
    elif config["env_mode"] == "server":
        date = mark_time[:8]
        hm = mark_time[8:10] + "-" + mark_time[10:12]
        dir = prefix + date + "/11/cloud_dugs_unet_3h/" + hm
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.png'):
                    time_str = file.split('.')[0]
                    time_obj = datetime.strptime(time_str, "%Y%m%d%H%M")
                    if time_obj > start_time_obj - timedelta(minutes=15):
                        res.append(os.path.abspath(os.path.join(root, file)))
        if len(res) == 0:
            raise HTTPException(
                status_code=404, detail=f"Image not found for the dir: {dir}")
        res.sort()
        mark_time_obj = datetime.strptime(mark_time, "%Y%m%d%H%M")
        if start_time_obj - mark_time_obj < timedelta(minutes=15):
            res.insert(0, "/data/ImageData/" +
                       start_time[:8]+"/11/real/" + align_time_15m(start_time) + ".png")
    for path in res:
        if not os.path.exists(path):
            print(f"{path} 不存在")
            res.remove(path)  # 从列表中移除不存在的路径
    return res


def generate_combined_map(image_files: list, speed, start_point, start_time: str, threshold=0, safety_radius=5):
    """speed: 每分钟移动的像素格子数"""

    start_time_obj = datetime.strptime(start_time, "%Y%m%d%H%M")
    # 读取第一个图像以获取地图大小
    sample_img = np.array(Image.open(image_files[0]).convert('L'))

    map_shape = sample_img.shape  # 获取地图尺寸

    # 初始化最终的综合障碍物地图
    combined_map = np.zeros(map_shape, dtype=np.uint8)

    # 预计算每个像素点到起点的距离图
    # np.indices 返回的数组 shape 为 (2, height, width)
    indices = np.indices(map_shape)

    # indices[0] 是 x 坐标，indices[1] 是 y 坐标
    distance_map = np.sqrt(
        (indices[0] - start_point[0]) ** 2 + (indices[1] - start_point[1]) ** 2)

    # 逐个时间步处理图像
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image_path}")

        # 计算当前时间步的半径范围
        time_str = re.search(r'(\d{12})(?=\.png)', image_path).group(0)
        time_obj = datetime.strptime(time_str, "%Y%m%d%H%M")
        min_radius = (time_obj + timedelta(minutes=15) -
                      start_time_obj).total_seconds() / 60 * speed
        # 读取并处理图像
        t = datetime.strptime(os.path.basename(image_path)[:12], "%Y%m%d%H%M")
        gray_array = np.array(Image.open(image_path).convert('L'))
        maps[t] = gray_array
        # 二值化（注意：根据实际图像情况可能需要调整阈值）
        bin_map = (gray_array > threshold).astype(np.uint8)

        # 形态学膨胀，增加障碍物的安全边界
        bin_map = binary_dilation(bin_map, structure=np.ones(
            (safety_radius, safety_radius))).astype(np.uint8)

        # 构造 annulus 区域的布尔掩码（矢量化）
        annulus_mask = (distance_map >= min_radius)

        # 将二值图中为障碍物的部分（bin_map==1）与 annulus 区域进行逻辑与，
        # 同时更新综合地图（逻辑或操作，相当于合并所有时间步的障碍物）
        combined_map[(bin_map == 1) & annulus_mask] = 1

    return combined_map


def insert_intermediate_points(path, threshold_distance):
    new_path = [path[0]]  # 保持起点不变
    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        # 如果线段长度超过阈值，拆分成多个点
        if dist > threshold_distance:
            # 计算需要插入多少个点
            num_points = int(dist // threshold_distance)

            for j in range(1, num_points):  # 从1开始，避免重复加入原始的起点
                # 按比例计算插入点坐标
                t = j * threshold_distance / dist
                new_point = [p1[0] + t * (p2[0] - p1[0]),
                             p1[1] + t * (p2[1] - p1[1])]
                new_path.append(new_point)

        # 将当前的终点添加到路径中
        new_path.append(p2)

    return new_path


def get_wh(image_path: str):
    img = Image.open(image_path)
    return (img.width, img.height)


def pos2pix(lat, lon):
    # 找到距离最近的像素点
    diff = np.sqrt((lookup_table[:, :, 0] - lat) **
                   2 + (lookup_table[:, :, 1] - lon) ** 2)
    i, j = np.unravel_index(np.argmin(diff), diff.shape)
    logging.info(f"(lat {lat}, lon {lon}) pix: row {i}, col {j})")
    return i, j


# 哈弗辛公式计算两点之间的距离（单位：公里）
def haversine(lat1, lon1, lat2, lon2):
    # 将角度转换为弧度
    R = 6371  # 地球半径，单位为公里
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * \
        math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 计算并返回距离
    return R * c


if __name__ == "__main__":
    start_time = "202411130715"
    image_files = get_images_path(start_time, mark_time="202411130700")
    combined_map = generate_combined_map(
        image_files, 6, (600, 600), start_time)
    plt.imshow(combined_map, cmap='gray')
    plt.savefig("temp.png")
