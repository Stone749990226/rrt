import bisect
import cProfile
from datetime import timedelta
import logging
import math
import time
from typing import Tuple
from matplotlib import patches, pyplot as plt
from matplotlib.widgets import Button
import yaml
from utils import Node, check_path_collision, generate_combined_map, get_images_path, has_collision, insert_intermediate_points
import numpy as np
from scipy.spatial import cKDTree
with open('rrt_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

animation = config["animation"]


class RRT:
    def __init__(self, width, height, step_size, end_lim, start: Node, end: Node, speed=6, use_config={
        "use_heuristic": False,
        "use_bidirectional": True,
        "use_adaptive_step": True,
            "collision_method": "bresenham"}) -> None:
        np.random.seed(42)
        self.t_iter_begin = time.time()
        # initial map & window
        self.height = height
        self.width = width
        # initial extend limitation and ede limitation
        self.step_size = step_size
        self.end_lim = end_lim
        self.speed = speed
        self.start = start
        self.end = end
        self.col_map = np.zeros([self.height, self.width])
        self.obstacle_kdtree = None
        self.use_config = use_config
        # node list
        # start_tree 和 end_tree 是分别从起点和终点开始生长的两棵 RRT* 树
        self.start_tree = [self.start]
        if self.use_config["use_bidirectional"]:
            self.end_tree = [self.end]

        self.less_long_path = np.inf
        self.last_path_length = np.inf
        self.path_all = []

        self.adaptive_params = {
            'density_radius': 100,
            'max_step_ratio': 1.5,
            'min_step_ratio': 0.5
        }

        if animation:
            self.fig, self.ax = plt.subplots(figsize=(12, 7))
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.invert_yaxis()
            self.ax.set_aspect('equal')  # 设置横纵坐标轴的单位长度相同
            self.mode = 'start'

            self.obs_scatter = self.ax.scatter(
                [], [], c='black', s=1, zorder=1)
            self.path_line, = self.ax.plot(
                [], [], color='lightcoral', linewidth=2, zorder=10)

            # 鼠标事件绑定
            self.cid = None  # 用于存储鼠标点击事件的ID

            # 按钮事件
            self.button_set_points = Button(
                plt.axes([0.45, 0.01, 0.1, 0.05]), 'SET START')
            self.button_set_points.on_clicked(
                self.on_button_set_points_clicked)

    def set_start(self, start_row, start_col):
        self.start = Node(start_row, start_col)
        self.start_tree = [self.start]

    def set_end(self, end_row, end_col):
        self.end = Node(end_row, end_col)
        self.end_tree = [self.end]

    def on_button_set_points_clicked(self, event):
        print("Click on the plot to set the start and end point.")

        # 如果已有点击事件监听，先移除
        if self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)

        # 设置新的鼠标点击事件监听
        self.cid = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_axes_click)

    def on_axes_click(self, event):
        """鼠标点击在画布上设置起点的回调函数"""
        if event.inaxes != self.ax:  # 如果点击的区域不是坐标轴区域
            return
        # 获取点击的坐标（取整）
        clicked_point = (round(event.ydata), round(event.xdata))
        if self.mode == 'start':
            self.set_start(clicked_point[0], clicked_point[1])
            print(f"Start point set at: {self.start}")
            self.ax.scatter(self.start.col, self.start.row,
                            c='red', label='Start', zorder=5, s=3)
            self.fig.canvas.draw()  # 刷新图形
            self.mode = 'end'
            print("Now, click to set the end point.")
        elif self.mode == 'end':
            self.set_end(clicked_point[0], clicked_point[1])
            print(f"End point set at: {self.end}")
            self.ax.scatter(self.end.col, self.end.row,
                            c='blue', label='End', zorder=5, s=3)
            self.fig.canvas.draw()
            self.mode = 'finished'
            # 移除鼠标点击事件监听，避免继续设置
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None
            self.search_path()

    def calculate_step_size(self, node):
        pos = np.array([node.row, node.col])
        count = self.obstacle_kdtree.query_ball_point(
            pos,
            r=self.adaptive_params['density_radius'],
            return_length=True
        )
        if count == 0:
            adaptive_step = self.step_size * \
                self.adaptive_params['max_step_ratio']
        elif 0 < count < 5:
            adaptive_step = self.step_size
        else:
            adaptive_step = self.step_size * \
                self.adaptive_params['min_step_ratio']
        return adaptive_step

    def calculate_density(self, node):
        """优化后的密度计算方法（使用KD树加速）"""
        if self.obstacle_kdtree is None:
            return 0.0

        # 使用圆形区域查询代替矩形区域
        pos = np.array([node.row, node.col])
        count = self.obstacle_kdtree.query_ball_point(
            pos,
            r=self.adaptive_params['density_radius'],
            return_length=True
        )
        return count
        # area = np.pi * (self.adaptive_params['density_radius']**2)
        # return count / max(area, 1)  # 防止除以零

    def set_col_map(self, binary_map):
        self.col_map = binary_map
        if self.use_config["use_adaptive_step"]:
            obstacle_points = np.column_stack(np.where(binary_map == 1))
            self.obstacle_kdtree = cKDTree(obstacle_points)

        if animation:
            # 获取障碍物的位置并绘制
            obstacle_positions = np.column_stack(np.where(binary_map == 1))
            self.obs_scatter.set_offsets(obstacle_positions[:, [1, 0]])
            self.fig.canvas.draw()
            plt.pause(0.01)  # 暂停0.1秒

    def point_in_obstacle(self, point):
        return self.col_map[point[0]][point[1]] == 1

    def informed_sample(self, cMax, cMin):
        if not self.use_config["use_heuristic"] or cMax == np.inf or abs(cMax - cMin) < 50:
            # 如果尚未找到路径，退化为全图随机采样
            new_r = np.random.uniform(0, self.height)
            new_c = np.random.uniform(0, self.width)
            return new_r, new_c

        # 椭圆参数计算
        a = cMax / 2.0
        b = math.sqrt(cMax**2 - cMin**2) / 2.0

        # 椭圆中心（中点）
        center_r = (self.start.row + self.end.row) / 2.0
        center_c = (self.start.col + self.end.col) / 2.0

        # 计算旋转角度（从起点指向终点的方向）
        dx = self.end.row - self.start.row
        dy = self.end.col - self.start.col
        theta = math.atan2(dy, dx)

        # 在单位圆内生成均匀分布的随机点
        r = np.random.random()  # sqrt确保均匀分布
        angle = 2 * math.pi * np.random.random()
        x = r * math.cos(angle)
        y = r * math.sin(angle)

        # 应用椭圆变换（旋转+缩放+平移）
        x_rot = x * a * math.cos(theta) - y * b * math.sin(theta)
        y_rot = x * a * math.sin(theta) + y * b * math.cos(theta)

        new_r = x_rot + center_r
        new_c = y_rot + center_c

        # 限制坐标在地图范围内
        new_r = np.clip(new_r, 0, self.height-1)
        new_c = np.clip(new_c, 0, self.width-1)

        return new_r, new_c

    def find_nearest(tree, target_r, target_c):
        # 使用 min 函数返回元组 (最小的节点, key的计算结果)
        nearest_node = min(tree, key=lambda node: (
            node.row - target_r)**2 + (node.col - target_c)**2)

        # 获取 key 的计算结果
        key_value = (nearest_node.row - target_r)**2 + \
            (nearest_node.col - target_c)**2

        return nearest_node, key_value

    def steer(self, tree: list[Node], new_r, new_c, tree_index):
        if not self.use_config["use_bidirectional"]:
            # 强制使用单向搜索逻辑
            tree_index = 1  # 始终操作start_tree
        # 找到最近的节点
        nearest_node = min(tree, key=lambda node: (
            node.row - new_r)**2 + (node.col - new_c)**2)

        # 计算自适应步长
        if self.use_config["use_adaptive_step"]:
            adaptive_step = self.calculate_step_size(nearest_node)
            # density = self.calculate_density(nearest_node)
            # min_step = self.step_size * self.adaptive_params['min_step_ratio']
            # adaptive_step = self.step_size * (1 - density) + min_step * density
            # adaptive_step = max(min_step, min(adaptive_step, self.step_size))
        else:
            adaptive_step = self.step_size
        distance = np.sqrt((nearest_node.row - new_r)**2 +
                           (nearest_node.col - new_c)**2)

        if distance <= adaptive_step:
            new_node = Node(new_r, new_c, nearest_node)
        else:
            ratio = adaptive_step / distance
            add_row = nearest_node.row + (new_r - nearest_node.row) * ratio
            add_col = nearest_node.col + (new_c - nearest_node.col) * ratio
            new_node = Node(add_row, add_col, nearest_node)

        # 保留原有碰撞检测和rewire逻辑
        if config["rewire"]:
            for node in tree:
                distance = np.sqrt((new_node.col-node.col)**2 +
                                   (new_node.row-node.row)**2)
                if distance < int(adaptive_step):  # 使用动态步长判断
                    if node == new_node.parent or node == self.start or node == self.end:
                        continue
                    if distance+new_node.distance < node.distance:
                        node.parent = new_node
                        node.distance = distance+new_node.distance

        if has_collision(self.col_map, nearest_node, new_node, method=self.use_config["collision_method"]):
            if tree_index == 2:
                self.start_tree, self.end_tree = self.end_tree, self.start_tree
            return None

        if animation:
            if tree_index == 1:
                color = 'gray'
            elif tree_index == 2:
                color = 'lightblue'
            rect = patches.Rectangle(
                (new_node.col - 2, new_node.row - 2), 4, 4,
                linewidth=1, edgecolor='green', facecolor='green'
            )
            self.ax.add_patch(rect)
            self.ax.plot([new_node.col, nearest_node.col],
                         [new_node.row, nearest_node.row], color=color, linewidth=1)
            self.fig.canvas.draw_idle()
            plt.pause(0.01)

        return new_node

    def spring(self, tree_index, informed_sample_flag=1):
        if not self.use_config["use_bidirectional"]:
            # 强制使用单向搜索逻辑
            tree_index = 1  # 始终操作start_tree
        # 生成新节点
        if informed_sample_flag:
            cMin = math.sqrt((self.start.row - self.end.row) **
                             2 + (self.start.col - self.end.col)**2)
            cMax = self.less_long_path
            new_r, new_c = self.informed_sample(cMax, cMin)
        else:
            new_r = int(self.height * np.random.rand())
            new_c = int(self.width * np.random.rand())

        # 双向RRT，交替扩展
        if tree_index == 2:
            self.start_tree, self.end_tree = self.end_tree, self.start_tree

        new_node = self.steer(self.start_tree, new_r, new_c, tree_index)
        if new_node is None:
            return False

        # add the new node into node list
        self.start_tree.append(new_node)

        if not self.use_config["use_bidirectional"]:
            # 单向模式下检查是否到达终点附近
            distance = (new_node.row - self.end.row)**2 + \
                (new_node.col - self.end.col)**2
            if distance <= self.end_lim**2:
                return True
            return False
        # the tree birthed from the end node;
        # 在第一颗树和新节点作用完成后，去考虑另一个树，从原来的树开始一直往new node连接，一直到撞到障碍物或者连接到new node（搜索结束）
        new_node2 = self.steer(self.end_tree, new_r, new_c, tree_index)

        if new_node2 is None:
            return False

        # add the new node into node list
        self.end_tree.append(new_node2)

        # 检查是否两棵树已连通
        # 如果走一步就到了新node，就直接退出了
        if new_node2 == new_node:
            if tree_index == 2:
                self.start_tree, self.end_tree = self.end_tree, self.start_tree
            return True
        else:
            while True:
                distance = np.sqrt((new_node2.col - new_node.col)
                                   ** 2 + (new_node2.row - new_node.row) ** 2)
                # 生成 new_node3（介于 new_node2 和 new_node 之间的新节点）
                if distance <= self.step_size:
                    # 如果 distance 小于 step_size，直接连上 new_node
                    new_node3 = Node(new_node.row, new_node.col, new_node2)
                else:
                    # 否则，沿着 new_node2 → new_node 方向前进一步
                    add_row = (new_node.row - new_node2.row) * \
                        self.step_size / distance + new_node2.row
                    add_col = (new_node.col - new_node2.col) * \
                        self.step_size / distance + new_node2.col
                    new_node3 = Node(add_row, add_col, new_node2)

                # check collision the second time: whether the path is in the collision!
                if has_collision(self.col_map, new_node2, new_node3, method=self.use_config["collision_method"]):
                    if tree_index == 2:
                        self.start_tree, self.end_tree = self.end_tree, self.start_tree
                    return False

                if animation:
                    rect = patches.Rectangle(
                        # (x, y), 宽度, 高度
                        (new_node3.col - 2, new_node3.row - 2), 4, 4,
                        linewidth=1, edgecolor='green', facecolor='green'
                    )
                    self.ax.add_patch(rect)
                    # 创建直线
                    self.ax.plot([new_node2.col, new_node3.col], [
                        new_node2.row, new_node3.row], color='lightblue', linewidth=1)
                    self.fig.canvas.draw()

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
    def extend(self, informed_sample_flag=0):
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
            if abs(self.last_path_length - self.less_long_path) < config["path_len_diff"] and len(self.path_all) > 1 and self.last_path_length != self.less_long_path \
                    or \
                    now-self.t_iter_begin > config["max_iter_time"] and len(self.path_all) > 0:
                self.is_success = False
                print("当前算法已经收敛了")
                return 0
            # if now-self.t_s>0.5 and len(self.path_all)>0:
            #     self.is_success=False
            #     return 0
            # consistently spring up new node until meet end requirement
            # spring the tree first which has less nodes
            # 如果 start_tree（从起点生长的树）的节点数量 小于等于 end_tree（从终点生长的树），则扩展 start_tree。否则，扩展 end_tree。
            if self.use_config["use_bidirectional"] and len(self.start_tree) <= len(self.end_tree):
                is_success = self.spring(1, informed_sample_flag)
            else:
                is_success = self.spring(2, informed_sample_flag)
            if is_success:
                temp = self.end_limitation()
                if temp != False:
                    self.path = self.results(temp)
                    break

        if animation:
            self.ax.plot([temp[0].col, temp[1].col], [temp[0].row,
                                                      temp[1].row], color='black', linewidth=1)
            self.fig.canvas.draw()
        num = len(self.path) - 2
        print('there are %d nodes betweeen start and end' % num)
        # print(self.path)
        self.path_length = 0
        for i in range(len(self.path) - 1):
            self.path_length += np.sqrt(
                (self.path[i].row - self.path[i + 1].row) ** 2 + (self.path[i].col - self.path[i + 1].col) ** 2)
        print('Current path len:', self.path_length, end=', ')

        if self.path_length <= self.less_long_path:
            print('This path is better. Save!')
        else:
            print('This path is worse. Delete!')

        # t_e = time.time()
        # print('搜索时间为:', t_e - self.t_s)
        self.last_path_length = self.path_length
        # 如果新生成的路径长度小于原来的长度，则绘出
        if self.path_length <= self.less_long_path:
            self.less_long_path = self.path_length
            self.path_all.append(self.path)

            if animation:
                self.draw_path()

    def draw_path(self):
        x_values = [[self.path[i].col, self.path[i + 1].col]
                    for i in range(len(self.path) - 1)]
        y_values = [[self.path[i].row, self.path[i + 1].row]
                    for i in range(len(self.path) - 1)]
        # 绘制连接这些点的线
        self.path_line.set_data(x_values, y_values)

        # 刷新图形
        self.fig.canvas.draw()

    # end requirement,返回的是能连接两个tree，且使得总长度最小的两个点
    # 在 双向 RRT 算法中，两棵树扩展到一定程度后，需要合并形成完整路径。这个函数就是寻找两棵树之间的最佳连接点，使得最终路径最短
    # 计算 start → temp1 → temp2 → end 这条完整路径
    def end_limitation(self):
        if self.use_config["use_bidirectional"]:
            # t1,t2是两个可连接的节点
            t1 = None
            t2 = None
            path_all_length = np.inf
            # start_tree和end_tree是两个tree
            for temp1 in self.start_tree:
                for temp2 in self.end_tree:
                    dis = np.inf
                    if (temp1.row - temp2.row) ** 2 + (temp1.col - temp2.col) ** 2 <= self.step_size ** 2:
                        # calculate the length of all path
                        temp_node = temp1
                        dis = 0
                        while True:
                            if temp_node == self.start:
                                break
                            dis += np.sqrt(
                                (temp_node.row - temp_node.parent.row) ** 2 + (temp_node.col - temp_node.parent.col) ** 2)
                            temp_node = temp_node.parent
                        temp_node = temp2
                        while True:
                            if temp_node == self.end:
                                break
                            dis += np.sqrt(
                                (temp_node.row - temp_node.parent.row) ** 2 + (temp_node.col - temp_node.parent.col) ** 2)
                            temp_node = temp_node.parent
                        dis += np.sqrt((temp1.row - temp2.row) **
                                       2 + (temp1.col - temp2.col) ** 2)
                    if dis < path_all_length:
                        t1 = temp1
                        t2 = temp2
            if t1 == None:
                return False
            return t1, t2
        else:
            nearest_to_goal = min(self.start_tree,
                                  key=lambda n: (n.row - self.end.row)**2 + (n.col - self.end.col)**2)
            return nearest_to_goal, self.end  # 返回最近节点和终点

    def search_path(self, iternation=100):
        print(
            "*"*5, f"search path from start {self.start} to end {self.end}", "*"*5)
        if not has_collision(self.col_map, self.start, self.end):
            logging.info("起点和终点的连线没有障碍物，可以直接通行")
            self.path = [self.start, self.end]
            self.path_all = [[self.start, self.end]]
            if animation:
                self.draw_path()
        else:
            self.t_search_begin = time.time()
            self.t_iter_begin = time.time()
            self.extend()
            # 终止条件为迭代iternation次
            # 提前结束条件为：有成功路径且搜索时间超过1s/某次搜索的时间过长/路径长度收敛
            for i in range(iternation):
                if time.time()-self.t_search_begin > config["max_search_time"] and len(self.path_all) > 0:
                    break
                if self.is_success == False:  # 表示路径长度收敛了
                    break
                # time.sleep(1)
                self.t_iter_begin = time.time()
                # self.init_map()
                self.update_path()
                self.t_iter_end = time.time()
                print('iter %d : path' % (i+1), self.path_length,
                      'time cost: ', self.t_iter_end - self.t_iter_begin)
            print('最优路径长度为：', self.less_long_path)
            t_search_end = time.time()
            print('总时间为:', t_search_end - self.t_search_begin)
            # self.init_map()
        path_end = self.path_all[-1]
        self.path_final = []
        for i in path_end:
            self.path_final.append([i.row, i.col])
        self.path_final = insert_intermediate_points(
            self.path_final, self.speed * 15)
        if animation:
            x_vals = [point[0] for point in self.path_final]
            y_vals = [point[1] for point in self.path_final]
            print("绘制途经点")
            # 绘制途经点
            self.ax.scatter(y_vals, x_vals, color='red',
                            label='途经点', s=10, zorder=100)
            self.fig.canvas.draw()

        return self.path_final

    def optim_path(self, path):
        """路径后处理算法"""
        if len(path) < 3:
            return path

        optimized = [path[0]]  # 始终保留起点
        current_index = 0

        while current_index < len(path)-1:
            # 尝试连接尽可能远的节点
            farthest_safe = current_index + 1  # 至少保留下一个节点
            for check_index in range(len(path)-1, current_index, -1):
                if not has_collision(self.col_map, path[current_index], path[check_index], method=self.use_config["collision_method"]):
                    farthest_safe = check_index
                    break
            optimized.append(path[farthest_safe])
            current_index = farthest_safe

        return optimized

    # when make it, go back to find the relavently low cost path
    # 从 end_limitation 选出的两个连接点出发，回溯出一条完整的路径，并进行优化
    def results(self, temp_all):
        if self.use_config["use_bidirectional"]:
            # create the path list from start node to temp_all[0]
            temp = temp_all[0]
            res2 = []
            res2.append(temp)
            while temp != self.start:
                temp = temp.parent
                res2.append(temp)
            # reverse the results
            res = []
            l = len(res2) - 1
            for i in range(len(res2)):
                count = l - i
                res.append(res2[count])

            # create the path list from temp_all[1] to end node
            temp = temp_all[1]
            res.append(temp)
            while temp != self.end:
                temp = temp.parent
                res.append(temp)
            # return the full path
            res = self.optim_path(res)
            return res
        else:
            # 单向模式路径生成
            nearest_node, _ = temp_all
            path = []
            # 从最近节点回溯到起点
            node = nearest_node
            while node.parent:
                path.append(node)
                node = node.parent
            path.reverse()
            # 添加终点
            path.append(self.end)
            res = self.optim_path(path)
            return res

    # draw arcs to find the better path
    def update_path(self):
        # node list
        self.start_tree = [self.start]
        self.end_tree = [self.end]
        self.extend(informed_sample_flag=1)

    def print_path(self):
        if self.path is not None:
            print("[", end="")
            for point in self.path:
                print(point, end=",")
            print("]")


def test_rrt_with_config(n=20, configs=None):
    import cProfile
    # 默认测试配置
    if configs is None:
        configs = [
            {"name": "Baseline", "use_heuristic": False, "use_bidirectional": True,
                "use_adaptive_step": False, "collision_method": "bresenham"},
            {"name": "+Heuristic", "use_heuristic": False, "use_bidirectional": True,
                "use_adaptive_step": True, "collision_method": "bresenham"},
            {"name": "+AdaptiveStep", "use_heuristic": True, "use_bidirectional": True,
                "use_adaptive_step": True, "collision_method": "bresenham"},
            {"name": "+Bresenham", "use_heuristic": False, "use_bidirectional": True,
                "use_adaptive_step": False, "collision_method": "bresenham"},
            # {"name": "All Improvements", "use_heuristic": True,
            #     "use_bidirectional": True, "use_adaptive_step": True, "collision_method": "bresenham"}
        ]

    # 生成测试用例（所有配置共享同一组测试用例）
    np.random.seed(999)  # 固定随机种子
    test_cases = []
    map_generated = False
    col_map = None

    # 生成障碍物地图（所有测试用例使用同一地图）
    while not map_generated:
        try:
            start_time = "202411130728"
            mark_time = "2024111307015"
            png_paths = get_images_path(start_time, mark_time)
            col_map = generate_combined_map(
                png_paths, speed=6, start_point=(100, 100), start_time=start_time)
            map_generated = True
        except:
            pass

    # 生成有效的测试用例
    while len(test_cases) < n:
        start_r = np.random.randint(0, config["height"])
        start_c = np.random.randint(0, config["width"])
        end_r = np.random.randint(0, config["height"])
        end_c = np.random.randint(0, config["width"])

        # 有效性检查
        if (col_map[start_r][start_c] == 0 and
            col_map[end_r][end_c] == 0 and
                has_collision(col_map, Node(start_r, start_c), Node(end_r, end_c))):
            test_cases.append(((start_r, start_c), (end_r, end_c)))

    # 执行测试
    results = {}
    for cfg in configs:
        print(f"\n=== 正在测试配置：{cfg['name']} ===")
        success = 0
        total_time = 0
        path_lengths = []
        profiler = cProfile.Profile()
        profiler.enable()  # 开始性能分析
        for (start, end) in test_cases:
            start_node = Node(start[0], start[1])
            end_node = Node(end[0], end[1])

            # 初始化RRT
            rrt = RRT(
                width=config["width"],
                height=config["height"],
                step_size=config["step_size"],
                end_lim=config["end_lim"],
                start=start_node,
                end=end_node,
                use_config=cfg,
            )
            rrt.set_col_map(col_map)
            np.random.seed(42)  # 固定随机种子
            # 执行搜索
            try:
                start_time = time.time()
                path = rrt.search_path()
                elapsed = time.time() - start_time

                if path:
                    success += 1
                    total_time += elapsed
                    path_lengths.append(rrt.less_long_path)
            except Exception as e:
                print(f"测试失败：{str(e)}")
                raise RuntimeError
                continue
        profiler.disable()  # 停止性能分析
        profiler.print_stats(sort="time")  # 输出性能分析结果
        # 记录结果
        avg_time = total_time / success if success > 0 else 0
        avg_length = np.mean(path_lengths) if path_lengths else 0
        success_rate = success / len(test_cases)

        results[cfg['name']] = {
            "success_rate": success_rate,
            "avg_time": avg_time,
            "avg_length": avg_length
        }

    # 打印结果
    print("\n=== 测试结果汇总 ===")
    print("{:<20} {:<15} {:<15} {:<15}".format(
        "配置名称", "成功率", "平均时间(s)", "平均长度"))

    for name in ["Baseline", "+Heuristic", "+Bidirectional",
                 "+AdaptiveStep", "+Bresenham", "All Improvements"]:
        if name in results:
            data = results[name]
            print("{:<20} {:<15.2%} {:<15.2f} {:<15.2f}".format(
                name,
                data["success_rate"],
                data["avg_time"],
                data["avg_length"]
            ))

    return results


if __name__ == "__main__":
    # test_rrt_with_config(30)
    start_time = "202411130728"
    mark_time = "202411130715"

    speed = 4
    use_config = {
        "use_heuristic": False,
        "use_bidirectional": True,
        "use_adaptive_step": False,
        "collision_method": "bresenham"
    }
    rrt_agent = RRT(config["width"], config["height"],
                    config["step_size"], config["end_lim"], Node(149, 1604), Node(88, 1813), use_config=use_config)
    png_paths = get_images_path(start_time, mark_time)
    rrt_agent.set_col_map(generate_combined_map(
        png_paths, speed=speed, start_point=(149, 1604), start_time=start_time))
    # plt.show()
    # profiler = cProfile.Profile()
    # profiler.enable()  # 开始性能分析
    rrt_agent.search_path()
    # plt.show()
    # profiler.disable()
    # profiler.print_stats(sort="time")  # 输出性能分析结果
    # plt.pause(100)
    path = rrt_agent.path_final
    print(path)
    # check_path_collision(path=path, speed=speed,
    #                      start_time=start_time, animation_flag=animation)
