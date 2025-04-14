import cProfile
from contextlib import redirect_stdout
from datetime import timedelta
import logging
import math
import time
from typing import Tuple
from matplotlib import patches, pyplot as plt
from matplotlib.widgets import Button
from utils import check_path_collision, generate_combined_map, get_images_path, insert_intermediate_points
import numpy as np
from scipy.spatial import cKDTree

from config import GLOBAL_CONFIG
import utils


class AlgorithmConfig:
    def __init__(self, heuristic=False, bidirectional=True, adaptive_step=False, collision_method="bresenham"):
        self.heuristic = heuristic
        self.bidirectional = bidirectional
        self.adaptive_step = adaptive_step
        self.collision_method = collision_method


ALGORITHM_CONFIG = AlgorithmConfig()


class Node:
    """表示路径规划树中的节点，优化代价计算和父子关系维护"""

    __slots__ = ("row", "col", "_parent", "_children", "_distance")  # 内存优化

    def __init__(self, r=0, c=0, parent=None):
        """
        初始化节点
        :param r: 行坐标
        :param c: 列坐标
        :param parent: 父节点引用，默认为None表示根节点
        """
        self.row = r
        self.col = c
        self._parent = None
        self._children = []  # 子节点注册表
        self._distance = 0.0

        # 使用属性setter进行初始化
        self.parent = parent

    @property
    def parent(self):
        """父节点访问器"""
        return self._parent

    @parent.setter
    def parent(self, new_parent):
        """
        父节点修改器，自动维护树结构关系
        时间复杂度: O(1) 基本操作 / O(k) 子节点更新 (k为子树规模)
        """
        if self._parent is new_parent:
            return

        # 解除旧父节点关系
        if self._parent:
            self._parent._children.remove(self)

        # 建立新父节点关系
        self._parent = new_parent
        if new_parent:
            new_parent._children.append(self)

        # 触发代价更新
        self._update_distance(propagate=True)

    @property
    def distance(self):
        """节点到根节点的累积路径代价"""
        return self._distance

    def _update_distance(self, propagate=False):
        """
        更新节点路径代价，可选是否传播到子树
        :param propagate: 是否递归更新子节点
        """
        # 基例：根节点代价为0
        if self.parent is None:
            new_dist = 0.0
        else:
            # 增量式计算：父节点代价 + 局部欧氏距离
            new_dist = self.parent.distance + self._step_cost(self.parent)

        # 判断是否需要更新
        if abs(new_dist - self._distance) > 1e-9:  # 浮点精度容差
            self._distance = new_dist
            # 递归更新子节点
            if propagate:
                for child in self._children:
                    child._update_distance(propagate=True)

    def _step_cost(self, other: "Node") -> float:
        """计算与相邻节点的局部路径代价"""
        return np.hypot(self.row - other.row, self.col - other.col)

    def __str__(self):
        return f"({self.row}, {self.col})"

    def __repr__(self):
        return f"Node({self.row}, {self.col}, distance={self.distance:.2f})"


class RRT:
    def __init__(
        self,
        start: Node,
        end: Node,
        width=GLOBAL_CONFIG["width"],
        height=GLOBAL_CONFIG["height"],
        step_size=GLOBAL_CONFIG["step_size"],
        end_lim=GLOBAL_CONFIG["end_lim"],
        speed=6,
        animation=False,
    ) -> None:
        np.random.seed(42)
        self.t_iter_begin = time.time()
        self.height = height
        self.width = width
        self.step_size = step_size
        self.end_lim = end_lim
        self.speed = speed
        self.start = start
        self.end = end
        # 障碍物地图（0-1 numpy二维数组）
        self.col_map = None

        # start_tree 和 end_tree 是分别从起点和终点开始生长的两棵 RRT* 树
        if start:
            self.set_start(start.row, start.col)
        if end:
            self.set_end(end.row, end.col)
        if ALGORITHM_CONFIG.bidirectional:
            self.end_tree = [self.end]
        if ALGORITHM_CONFIG.adaptive_step:
            self.adaptive_params = {"density_radius": 250, "max_step_ratio": 1.3, "min_step_ratio": 0.5}
            self.obstacle_count_max = 1000
            self.obstacle_kdtree = None

        self.animation = animation
        if self.animation:
            self.fig, self.ax = plt.subplots(figsize=(12, 7))
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.invert_yaxis()
            self.ax.set_aspect("equal")  # 设置横纵坐标轴的单位长度相同
            self.mode = "start"

            self.obs_scatter = self.ax.scatter([], [], c="black", s=1, zorder=1)
            (self.path_line,) = self.ax.plot([], [], color="lightcoral", linewidth=2, zorder=10)

            # 鼠标事件绑定
            self.cid = None  # 用于存储鼠标点击事件的ID

            # 按钮事件
            self.button_set_points = Button(plt.axes([0.45, 0.01, 0.1, 0.05]), "SET START")
            self.button_set_points.on_clicked(self.on_button_set_points_clicked)

    def set_start(self, start_row, start_col):
        """设置起点"""
        self.start = Node(start_row, start_col)
        self.start_tree = [self.start]
        self.iter_num = 0

    def set_end(self, end_row, end_col):
        """设置终点"""
        self.end = Node(end_row, end_col)
        if ALGORITHM_CONFIG.bidirectional:
            self.end_tree = [self.end]
        if ALGORITHM_CONFIG.heuristic:
            self.cMin = math.sqrt((self.start.row - self.end.row) ** 2 + (self.start.col - self.end.col) ** 2)
            self.center = ((self.start.row + self.end.row) / 2.0, (self.start.col + self.end.col) / 2.0)
            dx = self.end.row - self.start.row
            dy = self.end.col - self.start.col
            theta = math.atan2(dy, dx)
            self.cos_theta = math.cos(theta)
            self.sin_theta = math.sin(theta)

    def on_button_set_points_clicked(self, event):
        print("Click on the plot to set the start and end point.")

        # 如果已有点击事件监听，先移除
        if self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)

        # 设置新的鼠标点击事件监听
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.on_axes_click)

    def on_axes_click(self, event):
        """鼠标点击在画布上设置起点的回调函数"""
        if event.inaxes != self.ax:  # 如果点击的区域不是坐标轴区域
            return
        # 获取点击的坐标（取整）
        clicked_point = (round(event.ydata), round(event.xdata))
        if self.mode == "start":
            self.set_start(clicked_point[0], clicked_point[1])
            print(f"Start point set at: {self.start}")
            self.ax.scatter(self.start.col, self.start.row, c="red", label="Start", zorder=5, s=3)
            self.fig.canvas.draw()  # 刷新图形
            self.mode = "end"
            print("Now, click to set the end point.")
        elif self.mode == "end":
            self.set_end(clicked_point[0], clicked_point[1])
            print(f"End point set at: {self.end}")
            self.ax.scatter(self.end.col, self.end.row, c="blue", label="End", zorder=5, s=3)
            self.fig.canvas.draw()
            self.mode = "finished"
            # 移除鼠标点击事件监听，避免继续设置
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None
            self.search_path()

    def calculate_step_size(self, node):
        """计算自适应步长"""
        # 查找给定点周围指定半径内的所有点的索引
        count = self.obstacle_kdtree.query_ball_point(np.array([node.row, node.col]), r=self.adaptive_params["density_radius"], return_length=True)
        # print("周围有 %d 个障碍物" % count)
        if count == 0:
            adaptive_step = self.step_size * self.adaptive_params["max_step_ratio"]
        elif 0 < count < self.obstacle_count_max:
            adaptive_step = self.step_size
        else:
            adaptive_step = self.step_size * self.adaptive_params["min_step_ratio"]
        return adaptive_step

    def set_col_map(self, binary_map):
        """设置障碍物地图"""
        self.col_map = binary_map
        if ALGORITHM_CONFIG.adaptive_step:
            obstacle_points = np.column_stack(np.where(binary_map == 1))
            self.obstacle_kdtree = cKDTree(obstacle_points)

        if self.animation:
            # 获取障碍物的位置并绘制
            obstacle_positions = np.column_stack(np.where(binary_map == 1))
            self.obs_scatter.set_offsets(obstacle_positions[:, [1, 0]])
            self.fig.canvas.draw()

    def point_in_obstacle(self, point):
        return self.col_map[point[0]][point[1]] == 1

    def random_sample(self):
        new_r = np.random.uniform(0, self.height)
        new_c = np.random.uniform(0, self.width)
        return new_r, new_c

    def sample(self, informed_sample_flag: bool):
        if not informed_sample_flag or not ALGORITHM_CONFIG.heuristic:
            return self.random_sample()
        cMax = self.less_long_path
        cMin = self.cMin
        if cMax == np.inf or abs(cMax - cMin) < 50:
            # 如果尚未找到路径，退化为全图随机采样
            return self.random_sample()

        # 椭圆参数计算
        a = cMax / 2.0
        b = math.sqrt(cMax**2 - cMin**2 + 1e-4) / 2.0

        # 椭圆中心（中点）
        (center_r, center_c) = self.center

        # 在单位圆内生成均匀分布的随机点
        r = np.random.random()  # sqrt确保均匀分布
        angle = 2 * math.pi * np.random.random()
        x = r * math.cos(angle)
        y = r * math.sin(angle)

        # 应用椭圆变换（旋转+缩放+平移）
        x_rot = x * a * self.cos_theta - y * b * self.sin_theta
        y_rot = x * a * self.sin_theta + y * b * self.cos_theta

        new_r = x_rot + center_r
        new_c = y_rot + center_c

        # 限制坐标在地图范围内
        new_r = np.clip(new_r, 0, self.height - 1)
        new_c = np.clip(new_c, 0, self.width - 1)

        return new_r, new_c

    def steer(self, tree: list[Node], new_r, new_c, tree_index):
        """扩展一个节点，如果新生成的节点到现在的树的最近节点的连线被障碍物阻挡，则返回None，否则返回新生成的节点"""
        if not ALGORITHM_CONFIG.bidirectional:
            # 强制使用单向搜索逻辑
            tree_index = 1  # 始终操作start_tree
        # 找到最近的节点
        nearest_node = min(tree, key=lambda node: (node.row - new_r) ** 2 + (node.col - new_c) ** 2)

        # 计算自适应步长
        if ALGORITHM_CONFIG.adaptive_step:
            adaptive_step = self.calculate_step_size(nearest_node)
        else:
            adaptive_step = self.step_size

        # 新节点距离nearest_node不超过步长，如果生成的随机节点超过步长则在线段上截取步长
        distance = math.sqrt((nearest_node.row - new_r) ** 2 + (nearest_node.col - new_c) ** 2)
        if distance <= adaptive_step:
            new_node = Node(new_r, new_c, nearest_node)
        else:
            ratio = adaptive_step / distance
            add_row = nearest_node.row + (new_r - nearest_node.row) * ratio
            add_col = nearest_node.col + (new_c - nearest_node.col) * ratio
            new_node = Node(add_row, add_col, nearest_node)

        if GLOBAL_CONFIG["rewire"]:
            # 使用KD-Tree加速邻居搜索
            points = np.array([[n.row, n.col] for n in tree])
            kdtree = cKDTree(points)
            rewire_radius = 2.5 * adaptive_step  # 扩展搜索半径
            neighbors_indices = kdtree.query_ball_point([new_node.row, new_node.col], rewire_radius)

            for idx in neighbors_indices:
                node = tree[idx]
                # 跳过无效节点
                if node in (new_node.parent, self.start, self.end):
                    continue

                # 计算潜在新路径代价
                new_cost = new_node.distance + np.hypot(node.row - new_node.row, node.col - new_node.col)

                # 仅当满足以下条件时重布线
                if new_cost < node.distance - 1e-6 and not has_collision(  # 考虑浮点误差
                    self.col_map, new_node, node, method=ALGORITHM_CONFIG.collision_method
                ):

                    # 仅需设置parent，distance会自动更新
                    node.parent = new_node  # 这会触发_distance的递归更新

                    # 维护树结构双向连接
                    if hasattr(node, "_children"):
                        # 如果原父节点存在，通知其更新
                        original_parent = node.parent
                        if original_parent and hasattr(original_parent, "_children"):
                            original_parent._children.remove(node)
                        new_node._children.append(node)

        if has_collision(self.col_map, nearest_node, new_node, method=ALGORITHM_CONFIG.collision_method):
            if tree_index == 2:
                self.start_tree, self.end_tree = self.end_tree, self.start_tree
            return None

        if self.animation:
            if tree_index == 1:
                color = "gray"
            elif tree_index == 2:
                color = "lightblue"
            rect = patches.Rectangle((new_node.col - 2, new_node.row - 2), 4, 4, linewidth=1, edgecolor="green", facecolor="green")
            self.ax.add_patch(rect)
            self.ax.plot([new_node.col, nearest_node.col], [new_node.row, nearest_node.row], color=color, linewidth=1)
            plt.pause(0.001)

        return new_node

    def spring(self, tree_index, informed_sample_flag=True):
        self.iter_num += 1
        if not ALGORITHM_CONFIG.bidirectional:
            # 强制使用单向搜索逻辑
            tree_index = 1  # 始终操作start_tree

        new_r, new_c = self.sample(informed_sample_flag)

        # 双向RRT，交替扩展
        if tree_index == 2:
            self.start_tree, self.end_tree = self.end_tree, self.start_tree

        # Start tree先扩展
        new_node = self.steer(self.start_tree, new_r, new_c, tree_index)
        if new_node is None:
            return False
        self.start_tree.append(new_node)

        if not ALGORITHM_CONFIG.bidirectional:
            # 单向RRT模式下检查是否到达终点附近
            distance = (new_node.row - self.end.row) ** 2 + (new_node.col - self.end.col) ** 2
            if distance <= self.end_lim**2:
                return True
            return False

        # 扩展End tree，从原来的树开始一直往new node连接，一直到撞到障碍物或者连接到new node（搜索结束）
        new_node2 = self.steer(self.end_tree, new_r, new_c, tree_index)
        if new_node2 is None:
            return False
        self.end_tree.append(new_node2)

        # 检查是否两棵树已连通
        # 如果走一步就到了新node，就直接退出了
        if new_node2 == new_node:
            if tree_index == 2:
                self.start_tree, self.end_tree = self.end_tree, self.start_tree
            return True
        else:
            while True:
                distance = math.sqrt((new_node2.col - new_node.col) ** 2 + (new_node2.row - new_node.row) ** 2)
                if ALGORITHM_CONFIG.adaptive_step:
                    adaptive_step = self.calculate_step_size(new_node2)
                else:
                    adaptive_step = self.step_size
                # 生成 new_node3（介于 new_node2 和 new_node 之间的新节点）
                if distance <= adaptive_step:
                    # 如果 distance 小于 step_size，直接连上 new_node
                    new_node3 = Node(new_node.row, new_node.col, new_node2)
                else:
                    # 否则，沿着 new_node2 → new_node 方向前进一步
                    add_row = (new_node.row - new_node2.row) * adaptive_step / distance + new_node2.row
                    add_col = (new_node.col - new_node2.col) * adaptive_step / distance + new_node2.col
                    new_node3 = Node(add_row, add_col, new_node2)

                # check collision the second time: whether the path is in the collision!
                if has_collision(self.col_map, new_node2, new_node3, method=ALGORITHM_CONFIG.collision_method):
                    if tree_index == 2:
                        self.start_tree, self.end_tree = self.end_tree, self.start_tree
                    return False

                if self.animation:
                    rect = patches.Rectangle(
                        # (x, y), 宽度, 高度
                        (new_node3.col - 2, new_node3.row - 2),
                        4,
                        4,
                        linewidth=1,
                        edgecolor="green",
                        facecolor="green",
                    )
                    self.ax.add_patch(rect)
                    # 创建直线
                    self.ax.plot([new_node2.col, new_node3.col], [new_node2.row, new_node3.row], color="lightblue", linewidth=1)
                    self.fig.canvas.draw()
                    plt.pause(0.001)

                # add the new node into node list
                self.end_tree.append(new_node3)
                # 结束标志，同上
                if new_node3.row == new_node.row and new_node3.col == new_node.col:
                    if tree_index == 2:
                        self.start_tree, self.end_tree = self.end_tree, self.start_tree
                    return True
                # 更换new_node2，进行迭代
                new_node2 = new_node3

    # expend nodes, flag is to figure whether to limit the new springed node's position
    def extend(self, informed_sample_flag=False):
        # 如果extend的时间较大，大概率是因为此路径无法再优化了（椭圆内障碍物太多），这时直接退出就可以了;
        # 如果前后两次路径的差值小于1，则已收敛了
        self.is_success = True
        while True:
            now = time.time()
            # if now-self.t_s>10:S
            #     print('no path')
            #     exit()
            # 1. 如果当前路径和上次路径长度差异小于 path_len_diff 且路径已经收敛，则退出。
            # 2. 如果 算法运行时间超过 max_iter_time 秒，且至少已经找到一条路径，则退出
            if (
                abs(self.last_path_length - self.less_long_path) < GLOBAL_CONFIG["path_len_diff"]
                and len(self.path_all) > 1
                and self.last_path_length != self.less_long_path
                or now - self.t_iter_begin > GLOBAL_CONFIG["max_iter_time"]
                and len(self.path_all) > 0
            ):
                self.is_success = False
                print("当前算法已经收敛了")
                return 0
            # if now-self.t_s>0.5 and len(self.path_all)>0:
            #     self.is_success=False
            #     return 0
            # consistently spring up new node until meet end requirement
            # spring the tree first which has less nodes
            # 如果 start_tree（从起点生长的树）的节点数量 小于等于 end_tree（从终点生长的树），则扩展 start_tree。否则，扩展 end_tree。
            if ALGORITHM_CONFIG.bidirectional and len(self.start_tree) <= len(self.end_tree):
                is_success = self.spring(1, informed_sample_flag)
            else:
                is_success = self.spring(2, informed_sample_flag)
            if is_success:
                temp = self.end_limitation()
                if temp != False:
                    self.path = self.results(temp)
                    break

        if self.animation:
            self.ax.plot([temp[0].col, temp[1].col], [temp[0].row, temp[1].row], color="black", linewidth=1)
            self.fig.canvas.draw()
        num = len(self.path) - 2
        print("there are %d nodes betweeen start and end" % num)
        # print(self.path)
        self.path_length = 0
        for i in range(len(self.path) - 1):
            self.path_length += math.sqrt((self.path[i].row - self.path[i + 1].row) ** 2 + (self.path[i].col - self.path[i + 1].col) ** 2)
        print("Current path len:", self.path_length, end=", ")

        if self.path_length <= self.less_long_path:
            print("This path is better. Save!")
        else:
            print("This path is worse. Delete!")

        # t_e = time.time()
        # print('搜索时间为:', t_e - self.t_s)
        self.last_path_length = self.path_length
        # 如果新生成的路径长度小于原来的长度，则绘出
        if self.path_length <= self.less_long_path:
            self.less_long_path = self.path_length
            self.path_all.append(self.path)

            if self.animation:
                self.draw_path()

    def draw_path(self):
        x_values = [[self.path[i].col, self.path[i + 1].col] for i in range(len(self.path) - 1)]
        y_values = [[self.path[i].row, self.path[i + 1].row] for i in range(len(self.path) - 1)]
        # 绘制连接这些点的线
        self.path_line.set_data(x_values, y_values)

        # 刷新图形
        self.fig.canvas.draw()

    # end requirement,返回的是能连接两个tree，且使得总长度最小的两个点
    # 在 双向 RRT 算法中，两棵树扩展到一定程度后，需要合并形成完整路径。这个函数就是寻找两棵树之间的最佳连接点，使得最终路径最短
    # 计算 start → temp1 → temp2 → end 这条完整路径
    def end_limitation(self):
        if ALGORITHM_CONFIG.bidirectional:
            # t1,t2是两个可连接的节点
            t1 = None
            t2 = None
            path_all_length = np.inf
            # start_tree和end_tree是两个tree
            for temp1 in self.start_tree:
                for temp2 in self.end_tree:
                    dis = np.inf
                    if (temp1.row - temp2.row) ** 2 + (temp1.col - temp2.col) ** 2 <= self.step_size**2:
                        # calculate the length of all path
                        temp_node = temp1
                        dis = 0
                        while True:
                            if temp_node == self.start:
                                break
                            dis += math.sqrt((temp_node.row - temp_node.parent.row) ** 2 + (temp_node.col - temp_node.parent.col) ** 2)
                            temp_node = temp_node.parent
                        temp_node = temp2
                        while True:
                            if temp_node == self.end:
                                break
                            dis += math.sqrt((temp_node.row - temp_node.parent.row) ** 2 + (temp_node.col - temp_node.parent.col) ** 2)
                            temp_node = temp_node.parent
                        dis += math.sqrt((temp1.row - temp2.row) ** 2 + (temp1.col - temp2.col) ** 2)
                    if dis < path_all_length:
                        t1 = temp1
                        t2 = temp2
            if t1 == None:
                return False
            return t1, t2
        else:
            nearest_to_goal = min(self.start_tree, key=lambda n: (n.row - self.end.row) ** 2 + (n.col - self.end.col) ** 2)
            return nearest_to_goal, self.end  # 返回最近节点和终点

    def search_path(self, iternation=GLOBAL_CONFIG["iteration"]):
        # 截止本次iter的最短路径长度
        self.less_long_path = np.inf
        # 上一次路径长度，如果变化小于设定值，则认为路径收敛
        self.last_path_length = np.inf
        self.path_all = []
        print("*" * 5, f"search path from start {self.start} to end {self.end}", "*" * 5)
        if not has_collision(self.col_map, self.start, self.end):
            logging.info("起点和终点的连线没有障碍物，可以直接通行")
            self.path = [self.start, self.end]
            self.path_all = [[self.start, self.end]]
            if self.animation:
                self.draw_path()
        else:
            self.t_search_begin = time.time()
            self.t_iter_begin = time.time()
            self.extend()
            # 终止条件为迭代iternation次
            # 提前结束条件为：有成功路径且搜索时间超过1s/某次搜索的时间过长/路径长度收敛
            for i in range(iternation):
                if time.time() - self.t_search_begin > GLOBAL_CONFIG["max_search_time"] and len(self.path_all) > 0:
                    break
                if self.is_success == False:  # 表示路径长度收敛了
                    break
                self.t_iter_begin = time.time()
                self.extend(informed_sample_flag=True)
                self.t_iter_end = time.time()
                print("iter %d : path" % (i + 1), self.path_length, "time cost: ", self.t_iter_end - self.t_iter_begin)
            print("最优路径长度为：", self.less_long_path)
            t_search_end = time.time()
            print("总时间为:", t_search_end - self.t_search_begin)
            # self.init_map()
        path_end = self.path_all[-1]
        self.path_final = []
        for i in path_end:
            self.path_final.append([i.row, i.col])
        self.path_final = insert_intermediate_points(self.path_final, self.speed * 15)
        if self.animation:
            x_vals = [point[0] for point in self.path_final]
            y_vals = [point[1] for point in self.path_final]
            print("绘制途经点")
            # 绘制途经点
            self.ax.scatter(y_vals, x_vals, color="red", label="途经点", s=10, zorder=100)
            self.fig.canvas.draw()
        # print(self.iter_num, "次迭代")
        return self.path_final

    def optim_path(self, path):
        """路径后处理算法"""
        if len(path) < 3:
            return path

        optimized = [path[0]]  # 始终保留起点
        current_index = 0

        while current_index < len(path) - 1:
            # 尝试连接尽可能远的节点
            farthest_safe = current_index + 1  # 至少保留下一个节点
            for check_index in range(len(path) - 1, current_index, -1):
                if not has_collision(self.col_map, path[current_index], path[check_index], method=ALGORITHM_CONFIG.collision_method):
                    farthest_safe = check_index
                    break
            optimized.append(path[farthest_safe])
            current_index = farthest_safe

        return optimized

    # when make it, go back to find the relavently low cost path
    # 从 end_limitation 选出的两个连接点出发，回溯出一条完整的路径，并进行优化
    def results(self, temp_all):
        def trace_path(node: Node, target: Node) -> list[Node]:
            """回溯路径"""
            path = []
            current = node
            while current != target:
                path.append(current)
                current = current.parent
            path.append(target)  # 确保包含终点
            return path

        if ALGORITHM_CONFIG.bidirectional:
            path = trace_path(temp_all[0], self.start)[::-1] + trace_path(temp_all[1], self.end)
        else:
            # 单向模式路径生成
            nearest_node, _ = temp_all
            path = [self.end]
            # 从最近节点回溯到起点
            node = nearest_node
            while node.parent:
                path.append(node)
                node = node.parent
            path = path[::-1]
        return self.optim_path(path)
        # return path

    # draw arcs to find the better path
    def update_path(self):
        # node list
        self.start_tree = [self.start]
        if ALGORITHM_CONFIG.bidirectional:
            self.end_tree = [self.end]
        self.extend(informed_sample_flag=True)

    def print_path(self):
        if self.path is not None:
            print("[", end="")
            for point in self.path:
                print(point, end=",")
            print("]")


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


def test_rrt_with_config(n=20, configs=None):
    global ALGORITHM_CONFIG
    import cProfile
    import os
    import glob

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 匹配所有以 "result_" 开头的文件然后删除
    file_pattern = os.path.join(script_dir, "result_*")
    result_files = glob.glob(file_pattern)
    for file_path in result_files:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"删除文件: {file_path}")

    # 默认测试配置
    if configs is None:
        configs = [
            # ("Baseline", AlgorithmConfig(heuristic=False, bidirectional=False, adaptive_step=False, collision_method="discrete")),
            # ("Bidirectional", AlgorithmConfig(heuristic=False, bidirectional=True, adaptive_step=False, collision_method="discrete")),
            ("+Bresenham", AlgorithmConfig(heuristic=False, bidirectional=True, adaptive_step=False, collision_method="bresenham")),
            # ("+Heuristic", AlgorithmConfig(heuristic=True, bidirectional=True, adaptive_step=False, collision_method="bresenham")),
            # ("+AdaptiveStep", AlgorithmConfig(heuristic=False, bidirectional=True, adaptive_step=True, collision_method="bresenham")),
            # ("All", AlgorithmConfig(heuristic=True, bidirectional=True, adaptive_step=True, collision_method="bresenham")),
        ]

    # 生成测试用例（所有配置共享同一组测试用例）
    np.random.seed(0)
    test_cases = []
    col_map = utils.test_map()
    while len(test_cases) < n:
        start_r = np.random.randint(0, GLOBAL_CONFIG["height"])
        start_c = np.random.randint(0, GLOBAL_CONFIG["width"])
        end_r = np.random.randint(0, GLOBAL_CONFIG["height"])
        end_c = np.random.randint(0, GLOBAL_CONFIG["width"])
        # 过滤起点和终点就在障碍物上的测试用例
        if col_map[start_r][start_c] == 0 and col_map[end_r][end_c] == 0 and has_collision(col_map, Node(start_r, start_c), Node(end_r, end_c)):
            test_cases.append(((start_r, start_c), (end_r, end_c)))

    results = {}
    for cfg in configs:
        print(f"\n=== 正在测试配置：{cfg[0]} ===")
        success = 0
        total_time = 0
        path_lengths = []
        total_iter_num = 0

        ALGORITHM_CONFIG = cfg[1]
        # 开始性能分析
        profiler = cProfile.Profile()
        profiler.enable()
        rrt = RRT(
            start=None,
            end=None,
        )
        rrt.set_col_map(col_map)
        for start, end in test_cases:
            start_node = Node(start[0], start[1])
            end_node = Node(end[0], end[1])
            rrt.set_start(start_node.row, start_node.col)
            rrt.set_end(end_node.row, end_node.col)
            try:
                start_time = time.time()
                path = rrt.search_path()
                elapsed = time.time() - start_time

                if path:
                    success += 1
                    total_time += elapsed
                    path_lengths.append(rrt.less_long_path)
                    total_iter_num += rrt.iter_num
            except Exception as e:
                print(f"测试失败：{str(e)}")
                raise RuntimeError
                continue
        profiler.disable()  # 停止性能分析
        with open("result_" + cfg[0] + ".txt", "w") as f:
            with redirect_stdout(f):
                profiler.print_stats(sort="time")

        # 记录结果
        avg_time = total_time / success if success > 0 else 0
        avg_length = np.mean(path_lengths) if path_lengths else 0
        avg_iter_num = total_iter_num / success if success > 0 else 0
        success_rate = success / len(test_cases)

        results[cfg[0]] = {"success_rate": success_rate, "avg_time": avg_time, "avg_length": avg_length, "avg_iter_num": avg_iter_num}

    # 打印结果
    print("\n=== 测试结果汇总 ===")
    print("{:<20} {:<15} {:<15} {:<15} {:<10}".format("配置名称", "成功率", "平均时间(s)", "平均长度", "平均迭代次数"))

    for name in results.keys():
        data = results[name]
        print(
            "{:<20} {:<15.2%} {:<15.6f} {:<15.6f} {:<10}".format(
                name, data["success_rate"], data["avg_time"], data["avg_length"], data["avg_iter_num"]
            )
        )

    return results


if __name__ == "__main__":
    test_rrt_with_config(100)
    # start_time = "202411130715"
    # mark_time = "202411130715"
    # start = (149, 1604)
    # goal = (88, 1813)
    # speed = 4
    # ALGORITHM_CONFIG = AlgorithmConfig(heuristic=False, bidirectional=True, adaptive_step=True, collision_method="bresenham")
    # rrt_agent = RRT(Node(*start), Node(*goal), speed=speed, animation=GLOBAL_CONFIG["animation"])
    # png_paths = get_images_path(start_time, mark_time)
    # combined_map = generate_combined_map(png_paths, speed, start, start_time)
    # rrt_agent.set_col_map(combined_map)
    # plt.show()

    # profiler = cProfile.Profile()
    # profiler.enable()  # 开始性能分析
    # rrt_agent.search_path()
    # plt.show()
    # profiler.disable()
    # profiler.print_stats(sort="time")  # 输出性能分析结果
    # plt.pause(100)
    # path = rrt_agent.path_final
    # print(path)
