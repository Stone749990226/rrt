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
import time
from datetime import datetime, timedelta
from matplotlib import animation, pyplot as plt
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
    global maps

    start_time_obj = datetime.strptime(start_time, "%Y%m%d%H%M")

    map_shape = (config["height"], config["width"])  # 获取地图尺寸

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


def check_path_collision(path, speed, start_time: datetime | str, animation_flag=False):
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%Y%m%d%H%M")
    rounded_path = [(round(row), round(col)) for (row, col) in path]
    if len(rounded_path) < 2:
        return True

    segments = []
    for i in range(len(rounded_path)-1):
        start = rounded_path[i]
        end = rounded_path[i+1]
        d_row = end[0] - start[0]
        d_col = end[1] - start[1]
        distance = np.hypot(d_row, d_col)
        segments.append({
            'start': start,
            'end': end,
            'time': distance / speed,
            'distance': distance
        })
    total_time = sum(s['time'] for s in segments)

    map_times = sorted(maps.keys())
    elapsed = 0.0
    anim_data = []
    collision_info = None

    for seg in segments:
        seg_start_elapsed = elapsed
        seg_end_elapsed = seg_start_elapsed + seg['time']

        while elapsed < seg_end_elapsed:
            current_real_time = start_time + timedelta(minutes=elapsed)
            # 动态计算当前地图时间
            idx = bisect.bisect_right(map_times, current_real_time) - 1
            if idx < 0 or idx >= len(map_times):
                print("超出地图时间范围")
                return False
            current_map_time = map_times[idx]
            map_start = current_map_time
            map_end = map_start + timedelta(minutes=15)

            # 计算有效时间窗口
            window_start = max(map_start, start_time)
            window_end = min(map_end, start_time +
                             timedelta(minutes=seg_end_elapsed))
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
                    window_end = min(map_end, start_time +
                                     timedelta(minutes=seg_end_elapsed))
                    available = (
                        window_end - window_start).total_seconds() / 60
                    if available <= 1e-6:
                        print("时间窗口不足")
                        return False
                else:
                    elapsed += (map_end -
                                current_real_time).total_seconds() / 60
                    continue

            time_in_window = min(available, seg_end_elapsed - elapsed)
            ratio_start = (elapsed - seg_start_elapsed) / seg['time']
            ratio_end = ratio_start + time_in_window / seg['time']

            part_start = (
                int(seg['start'][0] + ratio_start *
                    (seg['end'][0] - seg['start'][0])),
                int(seg['start'][1] + ratio_start *
                    (seg['end'][1] - seg['start'][1]))
            )
            part_end = (
                int(seg['start'][0] + ratio_end *
                    (seg['end'][0] - seg['start'][0])),
                int(seg['start'][1] + ratio_end *
                    (seg['end'][1] - seg['start'][1]))
            )

            if bresenham_collision(maps[current_map_time], part_start, part_end):
                collision_time = elapsed + time_in_window
                collision_info = {
                    'map_time': current_map_time,
                    'position': part_end,
                    'collision_t': collision_time,
                    'last_safe': anim_data[-1]['end'] if anim_data else part_start
                }
                if animation_flag:
                    animate_path(anim_data, maps, rounded_path,
                                 start_time, collision_info)
                return False

            anim_data.append({
                'map_time': current_map_time,
                'start': part_start,
                'end': part_end,
                't_start': elapsed,
                't_end': elapsed + time_in_window
            })

            elapsed += time_in_window

    # 检查最后时间段
    if elapsed < total_time:
        last_map_time = anim_data[-1]['map_time'] if anim_data else None
        if last_map_time:
            last_map_end = last_map_time + timedelta(minutes=15)
            if (start_time + timedelta(minutes=elapsed)) >= last_map_end:
                print("路径未在最后地图有效期内完成")
                return False

    if animation_flag:
        animate_path(anim_data, maps, rounded_path, start_time)
    return True


def animate_path(animation_data, maps, path, start_time, collision_info=None):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    current_map = animation_data[0]['map_time'] if animation_data else list(maps.keys())[
        0]
    img = ax.imshow(maps[current_map], cmap='gray', origin='upper')

    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    ax.plot(path_cols, path_rows, 'r--', alpha=0.3)
    ax.scatter(path_cols, path_rows, c='red', s=20)

    traj_line, = ax.plot([], [], 'b-', lw=1.5)
    current_dot, = ax.plot([], [], 'bo', ms=8)
    collision_marker = ax.scatter(
        [], [], c='red', marker='x', s=100, visible=False)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8))

    end_time = collision_info['collision_t'] if collision_info else animation_data[-1]['t_end'] if animation_data else 0

    def update(frame):
        nonlocal current_map
        t = frame
        current_seg = None
        for seg in animation_data:
            if seg['t_start'] <= t <= seg['t_end']:
                current_seg = seg
                break
        if not current_seg:
            current_seg = animation_data[-1] if animation_data else None
            t = end_time

        if current_seg['map_time'] != current_map:
            current_map = current_seg['map_time']
            img.set_data(maps[current_map])

        traj_cols, traj_rows = [], []
        for seg in animation_data:
            if seg['t_end'] <= t:
                traj_cols.extend([seg['start'][1], seg['end'][1]])
                traj_rows.extend([seg['start'][0], seg['end'][0]])
            else:
                ratio = (t - seg['t_start']) / (seg['t_end'] - seg['t_start'])
                inter_col = seg['start'][1] + ratio * \
                    (seg['end'][1]-seg['start'][1])
                inter_row = seg['start'][0] + ratio * \
                    (seg['end'][0]-seg['start'][0])
                traj_cols.append(inter_col)
                traj_rows.append(inter_row)
                break

        traj_line.set_data(traj_cols, traj_rows)
        current_dot.set_data(
            [traj_cols[-1]], [traj_rows[-1]] if traj_cols else [])

        current_real_time = start_time + timedelta(minutes=t)
        time_text.set_text(current_real_time.strftime("%H:%M:%S"))

        if collision_info and t >= end_time:
            collision_marker.set_offsets([collision_info['position'][::-1]])
            collision_marker.set_visible(True)
            time_text.set_text(
                f'COLLISION!\n{current_real_time.strftime("%H:%M:%S")}')
            ax.plot([collision_info['last_safe'][1], collision_info['position'][1]],
                    [collision_info['last_safe'][0], collision_info['position'][0]], 'r-', lw=2)

        return img, traj_line, current_dot, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=int(end_time)+1, interval=50)
    ani.save('animation.gif', writer='pillow', fps=20)
    plt.show()


if __name__ == "__main__":
    # start_time = "202411130715"
    # image_files = get_images_path(start_time, mark_time="202411130700")
    # combined_map = generate_combined_map(
    #     image_files, 6, (600, 600), start_time)
    # plt.imshow(combined_map, cmap='gray')
    # plt.savefig("temp.png")
    start_time = "202411130728"
    mark_time = "2024111307015"
    speed = 6
    generate_combined_map(
        get_images_path(start_time, mark_time), speed=speed, start_point=(100, 100), start_time=start_time)
    path = [(784, 1203), (824.5941418531968, 1398.0988815480077),
            (822.041420065192, 1495.4286774299896), (857, 1596), (886, 1722),]
    speed = 10
    start_time = datetime.strptime(start_time, "%Y%m%d%H%M")
    result = check_path_collision(
        path, speed, start_time, maps, animation_flag=True)
    print("Path safe:", result)
