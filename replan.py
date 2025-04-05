"""
人工势场寻路算法实现
最基本的人工势场，存在目标点不可达及局部最小点问题
"""
import matplotlib.pyplot as plt
import math
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import time

from utils import test_map, timer


import math


class Vector2d:
    """
    2维向量，支持加减乘除，使用属性延迟计算提高性能
    """

    __slots__ = ('_deltaX', '_deltaY', '_length', '_direction')  # 减少内存占用

    def __init__(self, x, y):
        self._deltaX = x
        self._deltaY = y
        self._length = None
        self._direction = None

    @property
    def deltaX(self):
        """x分量只读属性"""
        return self._deltaX

    @property
    def deltaY(self):
        """y分量只读属性"""
        return self._deltaY

    @property
    def length(self):
        """向量长度，首次访问时计算并缓存"""
        if self._length is None:
            self._length = math.hypot(
                self._deltaX, self._deltaY)  # 比sqrt更优化的计算方式
        return self._length

    @property
    def direction(self):
        """单位向量方向，首次访问时计算并缓存"""
        if self._direction is None:
            length = self.length
            if length == 0:
                self._direction = (0.0, 0.0)  # 零向量保持tuple类型统一
            else:
                self._direction = (self._deltaX / length,
                                   self._deltaY / length)
        return self._direction

    def __add__(self, other):
        """向量加法，直接返回新实例避免中间计算"""
        return Vector2d(self._deltaX + other._deltaX, self._deltaY + other._deltaY)

    def __sub__(self, other):
        """向量减法"""
        return Vector2d(self._deltaX - other._deltaX, self._deltaY - other._deltaY)

    def __mul__(self, scalar):
        """标量右乘，直接返回新实例"""
        if not isinstance(scalar, (int, float)):
            raise TypeError("只能与数值类型相乘")
        return Vector2d(self._deltaX * scalar, self._deltaY * scalar)

    def __rmul__(self, scalar):
        """标量左乘"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        """标量除法"""
        if not isinstance(scalar, (int, float)):
            raise TypeError("只能与数值类型相除")
        return Vector2d(self._deltaX / scalar, self._deltaY / scalar)

    def __repr__(self):
        """优化字符串表示，避免重复计算"""
        return (f"Vector2d(dx={self._deltaX}, dy={self._deltaY}, "
                f"length={self.length:.2f}, dir={self.direction})")


class APF():
    """
    人工势场寻路
    """

    def __init__(self, start, goal, obstacles, k_att: float, k_rep: float, rr: float,
                 step_size: float, max_iters: int, goal_threshold: float, is_plot=False):
        """
        :param start: 起点
        :param goal: 终点
        :param obstacles: 障碍物列表，每个元素为Vector2d对象
        :param k_att: 引力系数
        :param k_rep: 斥力系数
        :param rr: 斥力作用范围
        :param step_size: 步长
        :param max_iters: 最大迭代次数
        :param goal_threshold: 离目标点小于此值即认为到达目标点
        :param is_plot: 是否绘图
        """
        self.start = Vector2d(start[0], start[1])
        self.current_pos = Vector2d(start[0], start[1])
        self.goal = Vector2d(goal[0], goal[1])
        self.obstacles = [Vector2d(OB[0], OB[1]) for OB in obstacles]
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr  # 斥力作用范围
        self.step_size = step_size
        self.max_iters = max_iters
        self.iters = 0
        self.goal_threashold = goal_threshold
        self.path = list()
        self.is_path_plan_success = False
        self.is_plot = is_plot
        self.delta_t = 0.01

    def attractive(self):
        """
        引力计算
        :return: 引力
        """
        att = (self.goal - self.current_pos) * self.k_att  # 方向由机器人指向目标点
        return att

    def repulsion(self):
        """
        斥力计算
        :return: 斥力大小
        """
        rep = Vector2d(0, 0)  # 所有障碍物总斥力
        for obstacle in self.obstacles:
            # obstacle = Vector2d(0, 0)
            t_vec = self.current_pos - obstacle
            if (t_vec.length > self.rr):  # 超出障碍物斥力影响范围
                pass
            else:
                rep += Vector2d(t_vec.direction[0], t_vec.direction[1]) * self.k_rep * (
                    1.0 / t_vec.length - 1.0 / self.rr) / (t_vec.length ** 2)
        return rep

    def path_plan(self):
        """
        path plan
        :return:
        """
        while (self.iters < self.max_iters and (self.current_pos - self.goal).length > self.goal_threashold):
            f_vec = self.attractive() + self.repulsion()
            self.current_pos += Vector2d(
                f_vec.direction[0], f_vec.direction[1]) * self.step_size
            self.iters += 1
            self.path.append(
                [self.current_pos.deltaX, self.current_pos.deltaY])
            if self.is_plot:
                plt.plot(self.current_pos.deltaX,
                         self.current_pos.deltaY, '.b')
                plt.pause(self.delta_t)
        if (self.current_pos - self.goal).length <= self.goal_threashold:
            self.is_path_plan_success = True


"""
人工势场寻路算法实现
改进人工势场，解决不可达问题，仍存在局部最小点问题
"""


def check_vec_angle(v1: Vector2d, v2: Vector2d):
    v1_v2 = v1.deltaX * v2.deltaX + v1.deltaY * v2.deltaY
    angle = math.acos(v1_v2 / (v1.length * v2.length)) * 180 / math.pi
    return angle


class APF_Improved(APF):
    def __init__(self, start, goal, obstacles, k_att: float, k_rep: float, rr: float,
                 step_size: float, max_iters: int, goal_threshold: float, is_plot=False):
        self.start = Vector2d(start[0], start[1])
        self.current_pos = Vector2d(start[0], start[1])
        self.goal = Vector2d(goal[0], goal[1])
        self.obstacles = [Vector2d(OB[0], OB[1]) for OB in obstacles]
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr  # 斥力作用范围
        self.step_size = step_size
        self.max_iters = max_iters
        self.iters = 0
        self.goal_threashold = goal_threshold
        self.path = list()
        self.is_path_plan_success = False
        self.is_plot = is_plot
        self.delta_t = 0.01

    def repulsion(self):
        """
        斥力计算, 改进斥力函数, 解决不可达问题
        :return: 斥力大小
        """
        rep = Vector2d(0, 0)  # 所有障碍物总斥力
        for obstacle in self.obstacles:
            # obstacle = Vector2d(0, 0)
            obs_to_rob = self.current_pos - obstacle
            rob_to_goal = self.goal - self.current_pos
            if (obs_to_rob.length > self.rr):  # 超出障碍物斥力影响范围
                pass
            else:
                rep_1 = Vector2d(obs_to_rob.direction[0], obs_to_rob.direction[1]) * self.k_rep * (
                    1.0 / obs_to_rob.length - 1.0 / self.rr) / (obs_to_rob.length ** 2) * (rob_to_goal.length ** 2)
                rep_2 = Vector2d(rob_to_goal.direction[0], rob_to_goal.direction[1]) * self.k_rep * (
                    (1.0 / obs_to_rob.length - 1.0 / self.rr) ** 2) * rob_to_goal.length
                rep += (rep_1+rep_2)
        return rep


if __name__ == '__main__':
    # 相关参数设置
    k_att, k_rep = 1.0, 0.8
    rr = 3
    # 步长0.5寻路1000次用时4.37s, 步长0.1寻路1000次用时21s
    step_size, max_iters, goal_threashold = 0.2, 10000, .2
    step_size_ = 2

    # 设置、绘制起点终点
    start, goal = (0, 0), (15, 15)
    is_plot = True
    if is_plot:
        fig = plt.figure(figsize=(7, 7))
        subplot = fig.add_subplot(111)
        subplot.set_xlabel('X-distance: m')
        subplot.set_ylabel('Y-distance: m')
        subplot.plot(start[0], start[1], '*r')
        subplot.plot(goal[0], goal[1], '*r')
    # 障碍物设置及绘制
    obs = [[1, 4], [2, 4], [3, 3], [6, 1], [6, 7], [10, 6], [11, 12], [14, 14]]
    print('obstacles: {0}'.format(obs))

    if is_plot:
        for OB in obs:
            circle = Circle(xy=(OB[0], OB[1]), radius=rr, alpha=0.3)
            subplot.add_patch(circle)
            subplot.plot(OB[0], OB[1], 'xk')
    # t1 = time.time()
    # for i in range(1000):

    # path plan
    if is_plot:
        apf = APF_Improved(start, goal, obs, k_att, k_rep,
                           rr, step_size, max_iters, goal_threashold, is_plot)
    else:
        apf = APF_Improved(start, goal, obs, k_att, k_rep,
                           rr, step_size, max_iters, goal_threashold, is_plot)
    with timer():
        apf.path_plan()
    if apf.is_path_plan_success:
        path = apf.path
        path_ = []
        i = int(step_size_ / step_size)
        while (i < len(path)):
            path_.append(path[i])
            i += int(step_size_ / step_size)

        if path_[-1] != path[-1]:  # 添加最后一个点
            path_.append(path[-1])
        print('planed path points:{}'.format(path_))
        print('path plan success')
        if is_plot:
            px, py = [K[0] for K in path_], [K[1]
                                             for K in path_]  # 路径点x坐标列表, y坐标列表
            subplot.plot(px, py, '^k')
            plt.show()
    else:
        print('path plan failed')
    # t2 = time.time()
    # print('寻路1000次所用时间:{}, 寻路1次所用时间:{}'.format(t2-t1, (t2-t1)/1000))
