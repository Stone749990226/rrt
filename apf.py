import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.spatial import KDTree

from config import GLOBAL_CONFIG
from utils import test_map


class APF:
    """人工势场法路径规划 (NumPy优化版)"""

    def __init__(
        self, start, goal, obstacles, k_att=1.0, k_rep=0.9, rr=40, step_size=1, max_iters=10000, goal_threshold=1, is_plot=GLOBAL_CONFIG["animation"]
    ):
        """
        :param start: 起点 (row, col)
        :param goal: 目标点 (row, col)
        :param obstacles: 障碍物2d 0-1矩阵，self.obstacles会把它转换成 [[row1, col1], [row2, col2], ...]的形式
        :param rr: 斥力影响半径
        """
        # 参数转换
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)

        self.obstacles = np.argwhere(obstacles == 1)
        self.obstacle_tree = KDTree(self.obstacles) if len(self.obstacles) > 0 else None
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr
        self.step_size = step_size
        self.max_iters = max_iters
        self.goal_threshold = goal_threshold
        self.is_plot = is_plot

        # 运行状态
        self.current_pos = self.start.copy()
        self.path = np.empty((max_iters + 1, 2))
        self.path[0] = self.start
        self.iters = 0
        self.is_success = False

        # 可视化参数
        self.plot_interval = 50  # 绘图更新间隔
        self.delta_t = 0.01

        # 初始化可视化
        if self.is_plot:
            self.fig, self.ax = plt.subplots(figsize=(12, 7))
            self.ax.set_xlim(0, GLOBAL_CONFIG["width"])
            self.ax.set_ylim(0, GLOBAL_CONFIG["height"])
            self.ax.invert_yaxis()
            self.ax.set_aspect("equal")
            self.ax.plot(start[1], start[0], "bs")
            self.ax.plot(goal[1], goal[0], "gs")
            self.ax.scatter([x[1] for x in self.obstacles], [x[0] for x in self.obstacles], c="black", s=1, zorder=1)
            # for obs in self.obstacles:
            #     self.ax.add_patch(Circle(obs, radius=rr, alpha=0.3))
            #     self.ax.plot(*obs, 'xk')
            plt.show(block=False)

    def attractive_force(self):
        """引力计算"""
        delta = self.goal - self.current_pos
        distance = np.linalg.norm(delta)
        if distance == 0:
            return np.zeros(2)
        return self.k_att * delta

    def repulsive_force(self):
        """斥力计算 (KDTree优化版)"""
        if self.obstacle_tree is None or len(self.obstacles) == 0:
            return np.zeros(2)

        # KDTree范围查询
        indices = self.obstacle_tree.query_ball_point(self.current_pos, self.rr)
        if not indices:
            return np.zeros(2)

        near_obstacles = self.obstacles[indices]
        delta = self.current_pos - near_obstacles
        distances = np.linalg.norm(delta, axis=1)

        # 斥力计算
        with np.errstate(divide="ignore", invalid="ignore"):
            direction = delta / distances[:, None]
            magnitude = self.k_rep * (1.0 / distances - 1.0 / self.rr) / (distances**2)
            forces = direction * magnitude[:, None]

        return np.sum(forces, axis=0)

    def total_force(self):
        """合力计算"""
        f_att = self.attractive_force()
        f_rep = self.repulsive_force()
        return f_att + f_rep

    def check_goal(self):
        """检查是否到达目标点"""
        return np.linalg.norm(self.goal - self.current_pos) <= self.goal_threshold

    def path_plan(self):
        """执行路径规划"""
        for self.iters in range(1, self.max_iters + 1):
            if self.check_goal():
                self.is_success = True
                break

            # 计算运动方向
            force = self.total_force()
            norm = np.linalg.norm(force)
            if norm == 0:
                break  # 局部最小值
            direction = force / norm

            # 更新位置
            self.current_pos += direction * self.step_size
            self.path[self.iters] = self.current_pos

            # 可视化更新
            if self.is_plot and self.iters % self.plot_interval == 0:
                self.ax.plot(self.current_pos[1], self.current_pos[0], ".b", markersize=2)
                self.fig.canvas.draw_idle()
                plt.pause(self.delta_t)

        # 裁剪有效路径
        self.path = self.path[: self.iters]
        return self.is_success


class APF_Improved(APF):
    """改进版人工势场法 (解决目标不可达问题)"""

    def repulsive_force(self):
        """改进斥力计算 (KDTree优化版)"""
        if self.obstacle_tree is None or len(self.obstacles) == 0:
            return np.zeros(2)

        # KDTree范围查询
        indices = self.obstacle_tree.query_ball_point(self.current_pos, self.rr)
        if not indices:
            return np.zeros(2)

        near_obstacles = self.obstacles[indices]
        delta = self.current_pos - near_obstacles
        distances = np.linalg.norm(delta, axis=1)

        # 目标方向计算
        goal_direction = self.goal - self.current_pos
        goal_distance = np.linalg.norm(goal_direction)
        if goal_distance == 0:
            return np.zeros(2)
        goal_dir_norm = goal_direction / goal_distance

        # 斥力计算
        with np.errstate(divide="ignore", invalid="ignore"):
            # 第一部分斥力
            direction = delta / distances[:, None]
            mag_part1 = (1.0 / distances - 1.0 / self.rr) / (distances**2)
            part1 = direction * mag_part1[:, None] * goal_distance**2

            # 第二部分斥力
            mag_part2 = (1.0 / distances - 1.0 / self.rr) ** 2
            part2 = goal_dir_norm * mag_part2[:, None] * goal_distance

            # 合并斥力
            forces = (part1 + part2) * self.k_rep

        return np.sum(forces, axis=0)


# 测试用例
if __name__ == "__main__":
    # 参数设置(row, col)
    start = (149, 1604)
    goal = (1000, 71)

    # 创建路径规划器
    apf = APF_Improved(start=start, goal=goal, obstacles=test_map())

    # 执行路径规划
    start_time = time()
    success = apf.path_plan()
    print(f"规划耗时: {time()-start_time:.2f}s")

    if success:
        print("路径规划成功!")
        print(apf.path)
        if apf.is_plot:
            apf.ax.plot(apf.path[:, 1], apf.path[:, 0], "k-", lw=1)
            plt.show()
    else:
        print("路径规划失败!")
