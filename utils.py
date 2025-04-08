from contextlib import contextmanager
import time
import bisect
from datetime import datetime, timedelta
import os
from PIL import Image
from fastapi import HTTPException
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
from datetime import datetime, timedelta
from matplotlib import animation, pyplot as plt
from matplotlib import patches
from config import GLOBAL_CONFIG
from scipy.ndimage import binary_dilation
import re

maps = {}
lookup_table = np.load("cloud_look_up_table_v2.npy")


class Node:
    """each node has varieties:row,col,parent"""

    def __init__(self, r=0, c=0, f=None):
        self.row = r
        self.col = c
        self.parent = f
        self.distance = 0
        parent = self.parent
        # 如果路径发生了变动时（例如，节点可能会被修改或重连），循环计算distance才比较准确。
        while True:
            if parent == None:
                break
            self.distance += np.sqrt((r - parent.row) ** 2 + (c - parent.col) ** 2)
            r = parent.row
            c = parent.col
            parent = parent.parent

    def __str__(self):
        return f"({self.row}, {self.col})"


def align_time_15m(time_str: str):
    time_obj = datetime.strptime(time_str, "%Y%m%d%H%M")
    aligned_time = time_obj.replace(minute=(time_obj.minute // 15) * 15, second=0, microsecond=0)
    return aligned_time.strftime("%Y%m%d%H%M")


def get_images_path(start_time, mark_time):
    """/data/ImageData/20241206/11/cloud_dugs_unet_3h/16-45"""
    res = []
    start_time_obj = datetime.strptime(start_time, "%Y%m%d%H%M")
    if GLOBAL_CONFIG["env_mode"] == "local":
        for root, dirs, files in os.walk("./07-00"):
            for file in files:
                if file.endswith(".png"):
                    time_str = file.split(".")[0]
                    time_obj = datetime.strptime(time_str, "%Y%m%d%H%M")
                    if time_obj > start_time_obj - timedelta(minutes=15):
                        res.append(os.path.abspath(os.path.join(root, file)))
        if len(res) == 0:
            raise HTTPException(status_code=404, detail=f"Image not found")
        res.sort()
    elif GLOBAL_CONFIG["env_mode"] == "server":
        date = mark_time[:8]
        hm = mark_time[8:10] + "-" + mark_time[10:12]
        dir = GLOBAL_CONFIG["image_path"] + date + "/11/cloud_dugs_unet_3h/" + hm
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".png"):
                    time_str = file.split(".")[0]
                    time_obj = datetime.strptime(time_str, "%Y%m%d%H%M")
                    if time_obj > start_time_obj - timedelta(minutes=15):
                        res.append(os.path.abspath(os.path.join(root, file)))
        if len(res) == 0:
            raise HTTPException(status_code=404, detail=f"Image not found for the dir: {dir}")
        res.sort()
        mark_time_obj = datetime.strptime(mark_time, "%Y%m%d%H%M")
        if start_time_obj - mark_time_obj < timedelta(minutes=15):
            res.insert(
                0,
                "/data/ImageData/" + start_time[:8] + "/11/real/" + align_time_15m(start_time) + ".png",
            )
    for path in res:
        if not os.path.exists(path):
            print(f"{path} 不存在")
            res.remove(path)  # 从列表中移除不存在的路径
    return res


def generate_combined_map(
    image_files: list,
    speed: int,
    start_point,
    start_time: str,
    threshold=0,
    safety_radius=1,
):
    """speed: 每分钟移动的像素格子数"""
    global maps

    start_time_obj = datetime.strptime(start_time, "%Y%m%d%H%M")

    map_shape = (GLOBAL_CONFIG["height"], GLOBAL_CONFIG["width"])  # 获取地图尺寸

    # 初始化最终的综合障碍物地图
    combined_map = np.zeros(map_shape, dtype=np.uint8)

    # 预计算每个像素点到起点的距离图
    # np.indices 返回的数组 shape 为 (2, height, width)
    indices = np.indices(map_shape)

    # indices[0] 是 x 坐标，indices[1] 是 y 坐标
    distance_map = np.sqrt((indices[0] - start_point[0]) ** 2 + (indices[1] - start_point[1]) ** 2)

    # 逐个时间步处理图像
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image_path}")

        # 计算当前时间步的半径范围
        time_str = re.search(r"(\d{12})(?=\.png)", image_path).group(0)
        time_obj = datetime.strptime(time_str, "%Y%m%d%H%M")
        min_radius = (time_obj + timedelta(minutes=15) - start_time_obj).total_seconds() / 60 * speed
        # 读取并处理图像
        t = datetime.strptime(os.path.basename(image_path)[:12], "%Y%m%d%H%M")
        gray_array = np.array(Image.open(image_path).convert("L"))

        # 二值化（注意：根据实际图像情况可能需要调整阈值）
        bin_map = (gray_array > threshold).astype(np.uint8)
        maps[t] = bin_map
        # 构造 annulus 区域的布尔掩码（矢量化）
        annulus_mask = distance_map >= min_radius

        # 将二值图中为障碍物的部分（bin_map==1）与 annulus 区域进行逻辑与，
        # 同时更新综合地图（逻辑或操作，相当于合并所有时间步的障碍物）
        combined_map[(bin_map == 1) & annulus_mask] = 1

    def cross_structure(radius):
        size = 2 * radius + 1
        structure = np.zeros((size, size), dtype=bool)
        center = radius
        structure[center, :] = True  # 水平线
        structure[:, center] = True  # 垂直线
        return structure

    # 形态学膨胀，增加障碍物的安全边界
    combined_map = binary_dilation(combined_map, structure=cross_structure(safety_radius)).astype(np.uint8)
    return combined_map


def test_map():
    start_time = "202411130728"
    mark_time = "202411130715"

    speed = 4
    png_paths = get_images_path(start_time, mark_time)
    return generate_combined_map(png_paths, speed=speed, start_point=(149, 1604), start_time=start_time)


def insert_intermediate_points(path, threshold_distance):
    new_path = [path[0]]  # 保持起点不变
    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        # 如果线段长度超过阈值，拆分成多个点
        if dist > threshold_distance:
            # 计算需要插入多少个点
            num_points = int(dist // threshold_distance)

            for j in range(1, num_points):  # 从1开始，避免重复加入原始的起点
                # 按比例计算插入点坐标
                t = j * threshold_distance / dist
                new_point = [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])]
                new_path.append(new_point)

        # 将当前的终点添加到路径中
        new_path.append(p2)

    return new_path


def has_collision(col_map, node1: Node, node2: Node, method="bresenham") -> bool:
    """带方法选择的碰撞检测函数
    Parameters:
        method: 'bresenham' - 使用Bresenham算法（默认）
                'discrete'  - 使用离散点采样法
    """
    x0, y0 = int(node1.row), int(node1.col)
    x1, y1 = int(node2.row), int(node2.col)

    # 公共预处理：检查起点终点自身是否在障碍物
    if col_map[x0][y0] > 0 or col_map[x1][y1] > 0:
        return True

    if method == "discrete":
        # 离散点采样法实现
        dx = x1 - x0
        dy = y1 - y0
        distance = math.hypot(dx, dy)

        if distance == 0:
            return False

        # 动态计算采样步长（至少1像素）
        step_size = max(1, int(distance / 1000))
        steps = int(distance / step_size) + 1

        for i in range(steps + 1):
            ratio = i / steps
            x = x0 + dx * ratio
            y = y0 + dy * ratio
            # 四舍五入取整并确保在边界内
            xi = min(max(round(x), 0), col_map.shape[0] - 1)
            yi = min(max(round(y), 0), col_map.shape[1] - 1)
            if col_map[xi][yi] > 0:
                return True
        return False
    else:
        # Bresenham算法实现
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        current_x, current_y = x0, y0
        while True:
            if col_map[current_x][current_y] > 0:
                return True
            if current_x == x1 and current_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                current_x += sx
            if e2 < dx:
                err += dx
                current_y += sy
        return False


def get_wh(image_path: str):
    img = Image.open(image_path)
    return (img.width, img.height)


def pos2pix(lat, lon):
    # 找到距离最近的像素点
    diff = np.sqrt((lookup_table[:, :, 0] - lat) ** 2 + (lookup_table[:, :, 1] - lon) ** 2)
    i, j = np.unravel_index(np.argmin(diff), diff.shape)
    print(f"(lat {lat}, lon {lon}) pix: row {i}, col {j})")
    return i, j


# 哈弗辛公式计算两点之间的距离（单位：公里）
def haversine(lat1, lon1, lat2, lon2):
    # 将角度转换为弧度
    R = 6371  # 地球半径，单位为公里
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 计算并返回距离
    return R * c


def bresenham_collision(map_array, start, end) -> bool:
    # 使用 Bresenham 算法生成路径上的所有网格点，检查是否有障碍物
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    current_x, current_y = x0, y0

    while True:
        # 检查当前网格点是否碰撞
        if map_array[current_x][current_y] > 0:
            return True
        if current_x == x1 and current_y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            current_x += sx
        if e2 < dx:
            err += dx
            current_y += sy

    return False


def check_path_collision(path: list, speed: int, start_time: str, animation_flag=False) -> bool:
    """检查路径是否发生碰撞，如果发生返回true，否则返回发生碰撞的段的起点和终点"""
    start_time = datetime.strptime(start_time, "%Y%m%d%H%M")
    if len(path) < 2:
        return False
    # 先将路径转换成一个个段（segment），依次检测每个段上是否发生碰撞
    segments = []
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        distance = np.hypot((end[0] - start[0]), (end[1] - start[1]))
        segments.append({"start": start, "end": end, "time": distance / speed, "distance": distance})
    total_time = sum(s["time"] for s in segments)

    map_times = sorted(maps.keys())
    # 从起点出发开始计算的相对时间
    elapsed = 0.0
    anim_data = []
    collision_info = None
    # 依次遍历每一个段
    for i, seg in enumerate(segments):
        print(f"遍历第 {i} 段 {seg}")
        seg_start_elapsed = elapsed
        seg_end_elapsed = seg_start_elapsed + seg["time"]
        print(f"当前段开始时间: {seg_start_elapsed}, 结束时间: {seg_end_elapsed}")
        while elapsed < seg_end_elapsed - 1e-6:
            # 根据当前时间戳查找障碍物地图有效时间段
            current_real_time = start_time + timedelta(minutes=elapsed)
            idx = bisect.bisect_right(map_times, current_real_time) - 1
            if idx < 0 or idx >= len(map_times):
                print("超出地图时间范围")
                return False
            current_map_time = map_times[idx]
            map_start = current_map_time
            map_end = map_start + timedelta(minutes=15)

            # 计算当前障碍物地图内有效时间窗口
            # 取max是为了处理start_time为08:07, map_start为08:00的情况
            window_start = max(map_start, start_time)
            window_end = min(map_end, start_time + timedelta(minutes=seg_end_elapsed))
            available = (window_end - window_start).total_seconds() / 60

            if available <= 1e-6:
                if current_real_time >= map_end:
                    # 切换到下一个地图
                    idx += 1
                    if idx >= len(map_times):
                        print("超出地图时间范围")
                        return False
                    current_map_time = map_times[idx]
                    map_start = current_map_time
                    map_end = map_start + timedelta(minutes=15)
                    # 重新计算可用时间
                    window_start = max(map_start, start_time)
                    window_end = min(map_end, start_time + timedelta(minutes=seg_end_elapsed))
                    available = (window_end - window_start).total_seconds() / 60
                    if available <= 1e-6:
                        print("时间窗口不足")
                        return False
                else:
                    elapsed += (map_end - current_real_time).total_seconds() / 60
                    continue
            # 一个段可能比较长，跨过多个障碍物地图，所以要对段进行分割找到位于某一个障碍物地图的部分的起点和终点
            time_in_window = min(available, seg_end_elapsed - elapsed)
            ratio_start = (elapsed - seg_start_elapsed) / seg["time"]
            ratio_end = ratio_start + time_in_window / seg["time"]
            part_start = (
                seg["start"][0] + ratio_start * (seg["end"][0] - seg["start"][0]),
                seg["start"][1] + ratio_start * (seg["end"][1] - seg["start"][1]),
            )
            part_end = (
                seg["start"][0] + ratio_end * (seg["end"][0] - seg["start"][0]),
                seg["start"][1] + ratio_end * (seg["end"][1] - seg["start"][1]),
            )

            if bresenham_collision(maps[current_map_time], part_start, part_end):
                collision_time = elapsed + time_in_window
                collision_info = {
                    "map_time": current_map_time,
                    "position": part_end,
                    "collision_t": collision_time,
                    "last_safe": (anim_data[-1]["end"] if len(anim_data) > 0 else part_start),
                }
                if animation_flag:
                    animate_path(anim_data, maps, path, start_time, collision_info)
                print("路径发生碰撞")
                return (seg["start"], seg["end"], current_map_time)

            anim_data.append(
                {
                    "map_time": current_map_time,
                    "start": part_start,
                    "end": part_end,
                    "t_start": elapsed,
                    "t_end": elapsed + time_in_window,
                }
            )

            elapsed += time_in_window

    # 检查最后时间段
    if elapsed < total_time:
        last_map_time = anim_data[-1]["map_time"] if anim_data else None
        if last_map_time:
            last_map_end = last_map_time + timedelta(minutes=15)
            if (start_time + timedelta(minutes=elapsed)) >= last_map_end:
                print("路径未在最后地图有效期内完成")
                return False

    if animation_flag:
        animate_path(anim_data, maps, path, start_time)
    return False


def animate_path(animation_data, maps, path, start_time, collision_info=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_aspect("equal")
    current_map_time = animation_data[0]["map_time"] if animation_data else collision_info["map_time"]
    img = ax.imshow(maps[current_map_time], cmap="gray", origin="upper")

    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    ax.plot(path_cols, path_rows, "r--", alpha=0.3)
    ax.scatter(path_cols, path_rows, c="red", s=20)

    (traj_line,) = ax.plot([], [], "b-", lw=1.5)
    (current_dot,) = ax.plot([], [], "bo", ms=8)
    collision_marker = ax.scatter([], [], c="red", marker="x", s=100, visible=False)
    time_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.8))

    end_time = collision_info["collision_t"] if collision_info else animation_data[-1]["t_end"]

    def update(frame):
        nonlocal current_map_time
        t = frame
        current_part = None
        for part in animation_data:
            if part["t_start"] <= t <= part["t_end"]:
                current_part = part
                break
        if current_part is None:
            current_part = animation_data[-1] if animation_data else None
            t = end_time

        if current_part["map_time"] != current_map_time:
            current_map_time = current_part["map_time"]
            img.set_data(maps[current_map_time])

        traj_cols, traj_rows = [], []
        for part in animation_data:
            if part["t_end"] <= t:
                traj_cols.extend([part["start"][1], part["end"][1]])
                traj_rows.extend([part["start"][0], part["end"][0]])
            else:
                ratio = (t - part["t_start"]) / (part["t_end"] - part["t_start"])
                inter_col = part["start"][1] + ratio * (part["end"][1] - part["start"][1])
                inter_row = part["start"][0] + ratio * (part["end"][0] - part["start"][0])
                traj_cols.append(inter_col)
                traj_rows.append(inter_row)
                break

        traj_line.set_data(traj_cols, traj_rows)
        current_dot.set_data([traj_cols[-1]], [traj_rows[-1]] if traj_cols else [])

        current_real_time = start_time + timedelta(minutes=t)
        time_text.set_text(current_real_time.strftime("%H:%M:%S"))

        if collision_info and t >= end_time:
            collision_marker.set_offsets([collision_info["position"][::-1]])
            collision_marker.set_visible(True)
            time_text.set_text(f'COLLISION!\n{current_real_time.strftime("%H:%M:%S")}')
            ax.plot(
                [collision_info["last_safe"][1], collision_info["position"][1]],
                [collision_info["last_safe"][0], collision_info["position"][0]],
                "r-",
                lw=2,
            )

        return img, traj_line, current_dot, time_text

    ani = animation.FuncAnimation(fig, update, frames=int(end_time) + 1, interval=50)
    ani.save("animation.gif", writer="pillow", fps=20)
    plt.show()


@contextmanager
def timer():
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"耗时: {end - start:.4f} 秒")


if __name__ == "__main__":
    # start_time = "202411130715"
    # image_files = get_images_path(start_time, mark_time="202411130700")
    # combined_map = generate_combined_map(
    #     image_files, 6, (600, 600), start_time)
    # plt.imshow(combined_map, cmap='gray')
    # plt.savefig("temp.png")
    # start_time = "202411130728"
    # mark_time = "2024111307015"
    # speed = 6
    # generate_combined_map(
    #     get_images_path(start_time, mark_time), speed=speed, start_point=(100, 100), start_time=start_time)
    # path = [(784, 1203), (824.5941418531968, 1398.0988815480077),
    #         (822.041420065192, 1495.4286774299896), (857, 1596), (886, 1722),]
    # speed = 10
    # start_time = datetime.strptime(start_time, "%Y%m%d%H%M")
    # result = check_path_collision(
    #     path, speed, start_time, maps, animation_flag=True)
    # print("Path safe:", result)
    lat_min, lon_min = np.min(lookup_table, axis=(0, 1))  # 获取最小的经纬度
    lat_max, lon_max = np.max(lookup_table, axis=(0, 1))  # 获取最大的经纬度

    print(f"Left Bottom Corner: (lat: {lat_min}, lon: {lon_min})")
    print(f"Top Right Corner: (lat: {lat_max}, lon: {lon_max})")
