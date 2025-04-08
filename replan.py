from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from apf import APF_Improved
from rrt import RRT

from utils import Node, bresenham_collision, check_path_collision, generate_combined_map, get_images_path, maps


if __name__ == "__main__":
    start_time = "202411130715"
    mark_time = "202411130715"
    start = (838, 1306)
    goal = (926, 1630)
    speed = 4
    rrt_agent = RRT(Node(*start), Node(*goal), speed=speed, animation=False)
    png_paths = get_images_path(start_time, mark_time)
    combined_map = generate_combined_map(png_paths, speed, start, start_time)
    rrt_agent.set_col_map(combined_map)
    # plt.show()
    # profiler = cProfile.Profile()
    # profiler.enable()  # 开始性能分析
    rrt_agent.search_path()
    path = rrt_agent.path_final
    print(path)
    ret = check_path_collision(
        path, speed, start_time, animation_flag=False)
    if ret:
        print("collision")
        (collision_start, collision_end, map_time) = ret
        obstacles = maps[map_time]
        apf = APF_Improved(
            start=collision_start,
            goal=collision_end,
            obstacles=obstacles,
            is_plot=True
        )
        apf.path_plan()

        i = path.index(collision_start)
        path = np.array(path[:i] + list(apf.path) + path[i+2:])
        print(path)
        # if apf.is_plot:
        #     apf.ax.plot(path[:, 1], path[:, 0], 'k-', lw=1)
        #     plt.show()
    check_path_collision(
        path, speed, start_time, animation_flag=True)
