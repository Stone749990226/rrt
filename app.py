import copy
import random
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta
import cProfile
import requests
import cv2
from io import BytesIO
import scipy
import pytz
import os
import math
from scipy.spatial.transform import Rotation as Rot
import logging
from matplotlib import patches, pyplot as plt
from fastapi.middleware.cors import CORSMiddleware
from utils import generate_combined_map, get_wh, haversine, lookup_table, pos2pix, get_images_path
from rrt import node, rrt

from config import config
# 全局变量

animation = config["animation"]
maps = {}
speed = 6  # 每分钟能走多少像素格子
map_interval = 15  # 地图切换间隔为15分钟
threshold_distance = speed * map_interval

start_time = None
mark_time = None


logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
fh = logging.FileHandler(filename='./server.log')
formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)

ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)  # 将日志输出至屏幕
logger.addHandler(fh)  # 将日志输出至文件
# 创建 FastAPI 应用
app = FastAPI()

# 允许所有来源的跨域请求
app.add_middleware(
    CORSMiddleware,
    # 允许所有来源的跨域请求，你也可以设置为具体的域名来限制请求来源
    allow_origins=["*"],
    # 参数设置为True表示允许携带身份凭证，如cookies
    allow_credentials=True,
    # 表示允许所有HTTP方法的请求
    allow_methods=["*"],
    # 表示允许所有请求头
    allow_headers=["*"]
)

# 定义请求体模型


class Point(BaseModel):
    lat: float
    lon: float


class RequestBody(BaseModel):
    start: Point
    end: Point
    speed: float  # km/h
    time_step: int
    mark_time: str
    start_time: str
    threshold: float
    structure_size: int

# 定义返回值模型


class Waypoint(Point):
    reach_time: str  # 添加额外的类型字段


class Route(BaseModel):
    start_point: Point
    end_point: Point
    waypoints: List[Waypoint]


class Summary(BaseModel):
    # distance_pix: float
    distance_haversine: float
    estimated_time: float
    find_path: bool
    detail: str


class ResponseBody(BaseModel):
    route: Route
    summary: Summary

# 定义计算逻辑


def calculate_response(data: RequestBody) -> ResponseBody:
    global start_time, mark_time, speed
    # 需要将传入的km/h转换为km/min，由于图片一个像素点是4km，还要除以4
    speed = data.speed / 60 // 4
    row_start, col_start = pos2pix(data.start.lat, data.start.lon)
    row_goal, col_goal = pos2pix(data.end.lat, data.end.lon)
    start_time = datetime.strptime(
        data.start_time, "%Y-%m-%d %H:%M").strftime("%Y%m%d%H%M")
    mark_time = datetime.strptime(
        data.mark_time, "%Y-%m-%d %H:%M").strftime("%Y%m%d%H%M")
    png_paths = get_images_path(start_time, mark_time)

    rrt_agent = rrt(config["width"], config["height"], config["step_size"], config["end_lim"], node(
        row_start, col_start), node(row_goal, col_goal))
    rrt_agent.set_col_map(generate_combined_map(
        png_paths, speed=speed, start_point=(row_start, col_start), start_time=start_time))

    if rrt_agent.point_in_obstacle((row_start, col_start)) or rrt_agent.point_in_obstacle((row_goal, col_goal)):
        route = Route(
            start_point=data.start,
            end_point=data.end,
            waypoints=[],
        )
        summary = Summary(
            distance_haversine=0,
            estimated_time=0,
            find_path=False,
            detail="start or end point is in obstacle"
        )
        return ResponseBody(route, summary)
    path = rrt_agent.search_path()
    # profiler.disable()  # 停止性能分析
    # profiler.print_stats(sort="time")  # 输出性能分析结果

    logging.info(path)

    start_utc = datetime.strptime(data.start_time, "%Y-%m-%d %H:%M")
    start_utc = pytz.utc.localize(start_utc)  # 设置为UTC时区
    beijing_tz = pytz.timezone('Asia/Shanghai')
    waypoints = []
    time = 0
    total_km = 0
    lat = None
    lon = None
    for p in path:
        p[0], p[1] = int(p[0]), int(p[1])
        if lat is not None:
            total_km += haversine(lookup_table[p[0], p[1], 0],
                                  lookup_table[p[0], p[1], 1], lat, lon)
        lat = lookup_table[p[0], p[1], 0]
        lon = lookup_table[p[0], p[1], 1]
        time = total_km / data.speed
        end_utc = start_utc + timedelta(hours=time)
        end_beijing = end_utc.astimezone(beijing_tz)
        waypoints.append(
            Waypoint(lat=lat, lon=lon, reach_time=end_beijing.strftime("%Y-%m-%d %H:%M")))

    # 构造响应
    route = Route(
        start_point=data.start,
        end_point=data.end,
        waypoints=waypoints,
    )
    summary = Summary(
        # distance_pix=res["distance"] * 4,
        distance_haversine=total_km,
        estimated_time=total_km / data.speed,
        find_path=True,
        detail=""
    )
    return ResponseBody(route=route, summary=summary)


# 定义 POST 路由
@app.post("/api/route", response_model=ResponseBody)
async def calculate_route(request: RequestBody):
    response = calculate_response(request)
    return response

if __name__ == "__main__":
    request_data = {
        "start": {
            "lat": 18.187606552494625,
            "lon": 117.02636718750001
        },
        "end": {
            "lat": 11.781325296112277,
            "lon": 134.38476562500003
        },
        "start_time": "2024-11-13 07:15",
        "mark_time": "2024-11-13 07:00",
        "speed": 500,
        "time_step": 15,
        "threshold": 0,
        "structure_size": 5
    }

    print(calculate_response(RequestBody(**request_data)))
