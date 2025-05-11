from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from apf import APF_Improved
from rrt import RRT, Node

from utils import bresenham_collision, calculate_path_len, check_path_collision, generate_combined_map, get_images_path, maps, timer


def dynamic_test():
    start_time = "202410240445"
    mark_time = "202410240445"
    start = (150, 1554)
    goal = (88, 1813)
    speed = 6
    rrt_agent = RRT(Node(*start), Node(*goal), speed=speed, animation=True)
    png_paths = get_images_path(start_time, mark_time, local_image_path="./04-00")
    combined_map = generate_combined_map(png_paths, speed, start, start_time, safety_radius=15)
    rrt_agent.set_col_map(combined_map)
    rrt_agent.search_path()
    plt.show()
    check_path_collision(rrt_agent.path_final, speed, start_time, animation_flag=True)


def dynamic_test2():
    start_time = "202411130900"
    mark_time = "202411130845"

    start = (342, 76)
    goal = (112, 131)
    speed = 5
    rrt_agent = RRT(Node(*start), Node(*goal), speed=speed, animation=True)
    png_paths = get_images_path(start_time, mark_time)
    combined_map = generate_combined_map(png_paths, speed, start, start_time)
    rrt_agent.set_col_map(combined_map)
    # plt.show()
    # profiler = cProfile.Profile()
    # profiler.enable()  # 开始性能分析
    rrt_agent.search_path()
    path = rrt_agent.path_final

    print("原路径长度", calculate_path_len(path))

    while True:
        ret = check_path_collision(path, speed, start_time, animation_flag=True)

        if ret:
            print("collision")
            (collision_start, collision_end, map_time) = ret
            obstacles = maps[map_time]
            apf = APF_Improved(start=collision_start, goal=collision_end, obstacles=obstacles, animation_flag=False)
            with timer():
                apf.path_plan()

            i = path.index(collision_start)
            path = np.array(path[:i] + list(apf.path) + path[i + 2 :])
            print("APF重规划后路径长度", calculate_path_len(path))
        else:
            print("no collision")
            break


if __name__ == "__main__":
    dynamic_test()
    # dynamic_test2()
