from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta
import pytz
from scipy.spatial.transform import Rotation as Rot
import logging
from matplotlib import patches, pyplot as plt
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from utils import check_path_collision, generate_combined_map, get_wh, haversine, lookup_table, pos2pix, get_images_path
from rrt import Node, RRT
from pathlib import Path
import config
from config import GLOBAL_CONFIG
import argparse

parser = argparse.ArgumentParser(description="start of unicorn")
parser.add_argument("--app_host", default="0.0.0.0",
                    type=str, help="web服务器监听的ip，公网访问需要为0.0.0.0")
parser.add_argument("--app_port", default=8123, type=int, help="web服务器监听的端口")
parser.add_argument("--height", default=1060, help="image height", type=int)
parser.add_argument("--width", default=1824, help="image width", type=int)
parser.add_argument("--step_size", default=50,
                    help="RRT algorithm step size", type=int)
parser.add_argument("--end_lim", default=50,
                    help="RRT algorithm end limit", type=int)
parser.add_argument("--max_iter_time", default=10,
                    help="RRT algorithm max time cost in each iteration", type=int)
parser.add_argument("--max_search_time", default=20,
                    help="RRT algorithm max time cost during the whole search", type=int)
parser.add_argument("--path_len_diff", default=1, help="", type=int)
parser.add_argument("--animation", action="store_true",
                    help="whether show animation")
parser.add_argument("--rewire", action="store_true",
                    help="RRT algorithm use rewire(目前存在问题)")
parser.add_argument(
    "--env",
    choices=["local", "production"],  # 允许的值
    default="local",                  # 默认值
    help="指定运行环境（默认：local）"
)


maps = {}

start_time = None
mark_time = None


# 确保日志目录存在
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)  # 自动递归创建目录


def setup_logging():
    """初始化应用日志配置"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建统一格式
    formatter = logging.Formatter(
        "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
    )

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # 文件处理器
    fh = logging.FileHandler(filename=LOG_DIR / "server.log")
    fh.setFormatter(formatter)

    # 配置根日志
    logger.addHandler(ch)
    logger.addHandler(fh)

    # 特别配置 Uvicorn 日志处理器
    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_access = logging.getLogger("uvicorn.access")

    # 清除原有处理器
    for uv_logger in [uvicorn_error, uvicorn_access]:
        uv_logger.handlers.clear()
        uv_logger.addHandler(fh)  # 添加文件处理器
        uv_logger.addHandler(ch)  # 保留控制台输出
        uv_logger.propagate = False  # 禁止向上传播


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

    rrt_agent = RRT(config.GLOBAL_CONFIG["width"], config.GLOBAL_CONFIG["height"], config.GLOBAL_CONFIG["step_size"], config.GLOBAL_CONFIG["end_lim"], Node(
        row_start, col_start), Node(row_goal, col_goal))
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
        logging.error("start or end point is in obstacle")
        return ResponseBody(route=route, summary=summary)
    path = rrt_agent.search_path()
    if GLOBAL_CONFIG["animation"]:
        print(path)
        check_path_collision(path=path, speed=speed,
                             start_time=start_time, animation_flag=GLOBAL_CONFIG["animation"])
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
@app.post("/routing/route", response_model=ResponseBody)
async def calculate_route(request: RequestBody):
    response = calculate_response(request)
    return response

if __name__ == "__main__":
    setup_logging()
    args = parser.parse_args()

    config.set_config("height", args.height)
    config.set_config("width", args.width)
    config.set_config("step_size", args.step_size)
    config.set_config("end_lim", args.end_lim)
    config.set_config("max_iter_time", args.max_iter_time)
    config.set_config("max_search_time", args.max_search_time)
    config.set_config("path_len_diff", args.path_len_diff)
    config.set_config("animation", args.animation)
    config.set_config("rewire", args.rewire)
    logging.info(GLOBAL_CONFIG)
    uvicorn.run(
        app="main:app",
        host=args.app_host,
        port=args.app_port,
        log_level="debug"
    )
    # request_data = {
    #     "start": {
    #         "lat": 29.49698759653577,
    #         "lon": 99.7998046875
    #     },
    #     "end": {
    #         "lat": 12.297068292853817,
    #         "lon": 134.07714843750003
    #     },
    #     "start_time": "2024-11-13 10:00",
    #     "mark_time": "2024-11-13 07:00",
    #     "speed": 500,
    #     "time_step": 15,
    #     "threshold": 0,
    #     "structure_size": 5
    # }

    # print(calculate_response(RequestBody(**request_data)))
