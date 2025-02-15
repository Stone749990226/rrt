import bisect
from datetime import timedelta
import logging
import math
import time
from matplotlib import patches, pyplot as plt
from matplotlib.widgets import Button
import yaml
from utils import generate_combined_map, get_images_path, insert_intermediate_points
import numpy as np

with open('rrt_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

animation = config["animation"]


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
            self.distance += np.sqrt((r-parent.row)**2+(c-parent.col)**2)
            r = parent.row
            c = parent.col
            parent = parent.parent

    def __str__(self):
        return f"({self.row}, {self.col})"


class RRT:
    def __init__(self, width, height, step_size, end_lim, start: Node, end: Node, speed=6) -> None:
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
        self.map = np.zeros([self.height, self.width])
        self.col_map = np.zeros([self.height, self.width])
        self.Co = []

        # node list
        # start_tree 和 end_tree 是分别从起点和终点开始生长的两棵 RRT* 树
        self.start_tree = [self.start]
        self.end_tree = [self.end]

        self.less_long_path = np.inf
        self.last_path_length = np.inf
        self.path_all = []

        self.adaptive_params = {
            'goal_bias_base': 0.3,
            'density_radius': 7,
            'min_step_ratio': 0.2
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
        clicked_point = Node(round(event.ydata), round(event.xdata), None)
        if self.mode == 'start':
            self.start = clicked_point
            self.start_tree = [self.start]
            print(f"Start point set at: {self.start}")
            self.ax.scatter(self.start.col, self.start.row,
                            c='red', label='Start', zorder=5, s=3)
            self.fig.canvas.draw()  # 刷新图形
            self.mode = 'end'
            print("Now, click to set the end point.")
        elif self.mode == 'end':
            self.end = clicked_point
            self.end_tree = [self.end]
            print(f"End point set at: {self.end}")
            self.ax.scatter(self.end.col, self.end.row,
                            c='blue', label='End', zorder=5, s=3)
            self.fig.canvas.draw()
            self.mode = 'finished'
            # 移除鼠标点击事件监听，避免继续设置
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None
            self.search_path()

    def set_col_map(self, bin_map):
        self.col_map = bin_map

        if animation:
            # 获取障碍物的位置并绘制
            obstacle_positions = np.column_stack(np.where(bin_map == 1))
            self.obs_scatter.set_offsets(obstacle_positions[:, [1, 0]])
            self.fig.canvas.draw()
            plt.pause(0.1)  # 暂停0.1秒

    def has_collision(self, node1: Node, node2: Node, flag: int) -> bool:
        # 使用 Bresenham 算法生成路径上的所有网格点，检查是否有障碍物
        x0, y0 = int(node1.row), int(node1.col)
        x1, y1 = int(node2.row), int(node2.col)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        current_x, current_y = x0, y0

        while True:
            # 检查当前网格点是否碰撞
            if self.col_map[current_x][current_y] > 0:
                if flag == 2:
                    self.start_tree, self.end_tree = self.end_tree, self.start_tree
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

    def point_in_obstacle(self, point):
        return self.col_map[point[0]][point[1]] == 1

    def calculate_obstacle_density(self, row, col):
        """计算节点周围障碍物密度"""
        radius = self.adaptive_params['density_radius']
        min_row = max(0, int(row)-radius)
        max_row = min(self.height-1, int(row)+radius)
        min_col = max(0, int(col)-radius)
        max_col = min(self.width-1, int(col)+radius)

        obstacle_count = np.sum(
            self.col_map[min_row:max_row+1, min_col:max_col+1])
        area = (max_row - min_row + 1) * (max_col - min_col + 1)
        return obstacle_count / area if area > 0 else 0

    def get_adaptive_step(self, density):
        """根据障碍物密度计算自适应步长"""
        min_step = self.step_size * self.adaptive_params['min_step_ratio']
        return self.step_size * (1 - density) + min_step * density

    def sample_and_steer(self, flag, mk_dir_flag):
        """采样新节点并生成路径"""
        # 障碍物感知采样逻辑
        if mk_dir_flag:
            new_r, new_c = self.informed_sample()
        else:
            new_r, new_c = self.random_sample()

        # 寻找最近节点
        nearest_node = self.find_nearest_node(self.start_tree, new_r, new_c)

        # 生成新节点
        return self.steer(nearest_node, new_r, new_c)

    def spring(self, flag, mk_dir_flag=1):
        # 障碍物感知参数
        MAX_GOAL_BIAS = 0.7  # 最大目标偏置概率
        MIN_GOAL_BIAS = 0.1  # 最小目标偏置概率
        DENSITY_THRESHOLD = 0.3  # 密度阈值

        new_r = int(self.height * np.random.rand())
        new_c = int(self.width * np.random.rand())
        bias = 2
        # mk_dir_flag 控制是否进行受限区域采样（即是否要在 Informed RRT* 的椭圆范围内采样）TODO: 优化
        if mk_dir_flag:
            bias = 2
            while True:
                new_r = int(self.height * np.random.rand())
                new_c = int(self.width * np.random.rand())
                if np.sqrt((new_r - self.start.row)**2 + (new_c - self.start.col)**2) + \
                   np.sqrt((new_r - self.end.row)**2 + (new_c - self.end.col)**2) <= self.path_length + bias:
                    break
        else:
            new_r = int(self.height * np.random.rand())
            new_c = int(self.width * np.random.rand())

        # 双向RRT，交替扩展
        if flag == 2:
            self.start_tree, self.end_tree = self.end_tree, self.start_tree
        # "Near". find rule:only the distance
        # 遍历 start_tree 中所有节点，找到 欧几里得距离最近的节点 temp_node
        min_node = float('inf')
        temp_node = Node()
        for i in range(len(self.start_tree)):
            temp = self.start_tree[i]
            dis_r = temp.row - new_r
            dis_c = temp.col - new_c
            distance = dis_r ** 2 + dis_c ** 2

            if distance < min_node**2 and distance > 0:
                temp_node = temp
                min_node = distance

        # "Steer" and "Edge". link nodes
        distance = np.sqrt(min_node)

        if distance <= self.step_size:
            # 如果最近的节点与新节点的距离 小于步长，直接创建 new_node 连接 temp_node
            new_node = Node(new_r, new_c, temp_node)

        else:
            # 如果 大于步长，则沿着 temp_node → (new_r, new_c) 方向移动 step_size 进行扩展，创建 new_node
            add_row = (new_r - temp_node.row) * \
                self.step_size / distance + temp_node.row
            add_col = (new_c - temp_node.col) * \
                self.step_size / distance + temp_node.col
            new_node = Node(add_row, add_col, temp_node)

        # rewire
        '''
        for temp in self.start_tree:
            distance = np.sqrt((new_node.col-temp.col)**2 +
                               (new_node.row-temp.row)**2)
            if distance < int(self.step_size):
                if temp == new_node.parent or temp == self.start or temp == self.end:
                    continue
                if distance+new_node.distance < temp.distance:
                    temp.parent = new_node
                    temp.distance = distance+new_node.distance
        '''

        # check collision the second time: whether the path is in the collision!
        if self.has_collision(temp_node, new_node, flag):
            return False

        if animation:
            rect = patches.Rectangle(
                (new_node.col - 2, new_node.row - 2), 4, 4,  # (x, y), 宽度, 高度
                linewidth=1, edgecolor='green', facecolor='green'
            )
            self.ax.add_patch(rect)
            # 创建直线
            self.ax.plot([new_node.col, temp_node.col], [
                new_node.row, temp_node.row], color='gray', linewidth=1)
            self.fig.canvas.draw()
            plt.pause(0.01)

        # add the new node into node list
        self.start_tree .append(new_node)

        # the tree birthed from the end node;
        # 在第一颗树和新节点作用完成后，去考虑另一个树，从原来的树开始一直往new node连接，一直到撞到障碍物或者连接到new node（搜索结束）
        min_node = float('inf')
        temp_node = Node()
        for i in range(len(self.end_tree)):
            temp = self.end_tree[i]
            dis_r = temp.row - new_node.row
            dis_c = temp.col - new_node.col
            distance = dis_r ** 2 + dis_c ** 2

            if distance < min_node and distance > 0:
                temp_node = temp
                min_node = distance

        # "Steer" and "Edge". link nodes
        distance = np.sqrt(min_node)
        if distance <= self.step_size:
            new_node2 = Node(new_node.row, new_node.col, temp_node)
        else:
            add_row = (new_r - temp_node.row) * \
                self.step_size / distance + temp_node.row
            add_col = (new_c - temp_node.col) * \
                self.step_size / distance + temp_node.col
            new_node2 = Node(add_row, add_col, temp_node)

        # check collision: whether the path is in the collision!
        if self.has_collision(temp_node, new_node2, flag):
            return False

        if animation:
            rect = patches.Rectangle(
                (new_node.col - 2, new_node.row - 2), 4, 4,  # (x, y), 宽度, 高度
                linewidth=1, edgecolor='green', facecolor='green'
            )
            self.ax.add_patch(rect)
            # 创建直线
            self.ax.plot([new_node2.col, temp_node.col], [
                new_node2.row, temp_node.row], color='lightblue', linewidth=1)
            self.fig.canvas.draw()

        # add the new node into node list
        self.end_tree .append(new_node2)

        # 检查是否两棵树已连通
        # 如果走一步就到了新node，就直接退出了
        if new_node2 == new_node:
            if flag == 2:
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
                if self.has_collision(new_node2, new_node3, flag):
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
                self.end_tree .append(new_node3)
                # 结束标志，同上
                if new_node3.row == new_node.row and new_node3.col == new_node.col:
                    if flag == 2:
                        self.start_tree, self.end_tree = self.end_tree, self.start_tree
                    return True
                # 更换new_node2，进行迭代
                new_node2 = new_node3

    # expend nodes, flag is to figure whether to limit the new springed node's position
    def extend(self, flag=0):
        # 如果extend的时间较大，大概率是因为此路径无法再优化了（椭圆内障碍物太多），这时直接退出就可以了;
        # 如果前后两次路径的差值小于1，则已收敛了
        self.is_success = True
        while True:
            now = time.time()
            # if now-self.t_s>10:S
            #     print('no path')
            #     exit()
            # 1. 如果当前路径和上次路径长度差异小于 0.5 且路径已经收敛，则退出。
            # 2. 如果 算法运行时间超过 0.5 秒，且至少已经找到一条路径，则退出
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
            if len(self.start_tree) <= len(self.end_tree):
                is_success = self.spring(1, flag)
            else:
                is_success = self.spring(2, flag)
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
        print('路径上包含了%d个节点' % num)
        # print(self.path)
        self.path_length = 0
        # calculate 2a=path_length
        for i in range(len(self.path) - 1):
            self.path_length += np.sqrt(
                (self.path[i].row - self.path[i + 1].row) ** 2 + (self.path[i].col - self.path[i + 1].col) ** 2)
        print('当前路径长度为：', self.path_length, end=',')

        if self.path_length <= self.less_long_path:
            print('该路径优于上一次生成的路径，保留！')
        else:
            print('该路径次于上一次生成的路径，删除！')

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

    def search_path(self, iternation=100, max_search_time=10):
        if self.has_collision(self.start, self.end, 0) is False:
            logging.info("起点和终点的连线没有障碍物，可以直接通行")
            self.path = [[self.start, self.end]]
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
                if time.time()-self.t_search_begin > max_search_time and len(self.path_all) > 0:
                    break
                if self.is_success == False:  # 表示路径长度收敛了
                    break
                # time.sleep(1)
                self.t_iter_begin = time.time()
                # self.init_map()
                self.update_path()
                self.t_iter_end = time.time()
                print('第 %d 次迭代的路径长度为：' % (i+1), self.path_length,
                      '时间为：', self.t_iter_end - self.t_iter_begin)
            print('最优路径长度为：', self.less_long_path)
            t_t_search_end = time.time()
            print('总时间为:', t_t_search_end - self.t_search_begin)
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
                if not self.has_collision(path[current_index], path[check_index], 0):
                    farthest_safe = check_index
                    break
            optimized.append(path[farthest_safe])
            current_index = farthest_safe

        return optimized

    # when make it, go back to find the relavently low cost path
    # 从 end_limitation 选出的两个连接点出发，回溯出一条完整的路径，并进行优化
    def results(self, temp_all):
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

    # draw arcs to find the better path
    def update_path(self):
        # node list
        self.start_tree = []
        self.end_tree = []
        self.start_tree .append(self.start)
        self.end_tree .append(self.end)
        self.extend(flag=1)

    def print_path(self):
        if self.path is not None:
            print("[", end="")
            for point in self.path:
                print(point, end=",")
            print("]")


if __name__ == "__main__":
    start_time = "202411130728"
    mark_time = "2024111307015"
    speed = 6

    rrt_agent = RRT(config["width"], config["height"],
                    config["step_size"], config["end_lim"], None, None)
    png_paths = get_images_path(start_time, mark_time)
    rrt_agent.set_col_map(generate_combined_map(
        png_paths, speed=speed, start_point=(100, 100), start_time=start_time))
    plt.show()
    plt.pause(10000)
