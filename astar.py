"""
A_star 2D
@author: huiming zhou
"""

import matplotlib.pyplot as plt
import utils
import math
import heapq
from config import GLOBAL_CONFIG


class Env:

    def __init__(self):
        self.x_range = GLOBAL_CONFIG["width"]
        self.y_range = GLOBAL_CONFIG["height"]
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = {(i, j) for i, row in enumerate(utils.test_map()) for j, val in enumerate(row) if val == 1}

    def update_obs(self, map):
        self.obs = {(i, j) for i, row in enumerate(map) for j, val in enumerate(row) if val == 1}

    def obs_map(self):
        return self.obs


ENV = Env()


class AStar:
    """AStar set the cost + heuristics as the priority"""

    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = ENV  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        path = self.extract_path(self.PARENT)
        # path = self.optim_path(path)
        return path, self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN, (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

    # def optim_path(self, path):
    #     """路径后处理算法"""
    #     if len(path) < 3:
    #         return path

    #     optimized = [path[0]]  # 始终保留起点
    #     current_index = 0

    #     while current_index < len(path)-1:
    #         # 尝试连接尽可能远的节点
    #         farthest_safe = current_index + 1  # 至少保留下一个节点
    #         for check_index in range(len(path)-1, current_index, -1):
    #             if not utils.has_collision(self.col_map, path[current_index], path[check_index], method=ALGORITHM_CONFIG.collision_method):
    #                 farthest_safe = check_index
    #                 break
    #         optimized.append(path[farthest_safe])
    #         current_index = farthest_safe

    #     return optimized


class BidirectionalAStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = ENV  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN_fore = []  # OPEN set for forward searching
        self.OPEN_back = []  # OPEN set for backward searching
        self.CLOSED_fore = []  # CLOSED set for forward
        self.CLOSED_back = []  # CLOSED set for backward
        self.PARENT_fore = dict()  # recorded parent for forward
        self.PARENT_back = dict()  # recorded parent for backward
        self.g_fore = dict()  # cost to come for forward
        self.g_back = dict()  # cost to come for backward

    def init(self):
        """
        initialize parameters
        """

        self.g_fore[self.s_start] = 0.0
        self.g_fore[self.s_goal] = math.inf
        self.g_back[self.s_goal] = 0.0
        self.g_back[self.s_start] = math.inf
        self.PARENT_fore[self.s_start] = self.s_start
        self.PARENT_back[self.s_goal] = self.s_goal
        heapq.heappush(self.OPEN_fore, (self.f_value_fore(self.s_start), self.s_start))
        heapq.heappush(self.OPEN_back, (self.f_value_back(self.s_goal), self.s_goal))

    def searching(self):
        """
        Bidirectional A*
        :return: connected path, visited order of forward, visited order of backward
        """

        self.init()
        s_meet = self.s_start

        while self.OPEN_fore and self.OPEN_back:
            # solve foreward-search
            _, s_fore = heapq.heappop(self.OPEN_fore)

            if s_fore in self.PARENT_back:
                s_meet = s_fore
                break

            self.CLOSED_fore.append(s_fore)

            for s_n in self.get_neighbor(s_fore):
                new_cost = self.g_fore[s_fore] + self.cost(s_fore, s_n)

                if s_n not in self.g_fore:
                    self.g_fore[s_n] = math.inf

                if new_cost < self.g_fore[s_n]:
                    self.g_fore[s_n] = new_cost
                    self.PARENT_fore[s_n] = s_fore
                    heapq.heappush(self.OPEN_fore, (self.f_value_fore(s_n), s_n))

            # solve backward-search
            _, s_back = heapq.heappop(self.OPEN_back)

            if s_back in self.PARENT_fore:
                s_meet = s_back
                break

            self.CLOSED_back.append(s_back)

            for s_n in self.get_neighbor(s_back):
                new_cost = self.g_back[s_back] + self.cost(s_back, s_n)

                if s_n not in self.g_back:
                    self.g_back[s_n] = math.inf

                if new_cost < self.g_back[s_n]:
                    self.g_back[s_n] = new_cost
                    self.PARENT_back[s_n] = s_back
                    heapq.heappush(self.OPEN_back, (self.f_value_back(s_n), s_n))

        return self.extract_path(s_meet), self.CLOSED_fore, self.CLOSED_back

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def extract_path(self, s_meet):
        """
        extract path from start and goal
        :param s_meet: meet point of bi-direction a*
        :return: path
        """

        # extract path for foreward part
        path_fore = [s_meet]
        s = s_meet

        while True:
            s = self.PARENT_fore[s]
            path_fore.append(s)
            if s == self.s_start:
                break

        # extract path for backward part
        path_back = []
        s = s_meet

        while True:
            s = self.PARENT_back[s]
            path_back.append(s)
            if s == self.s_goal:
                break

        return list(reversed(path_fore)) + list(path_back)

    def f_value_fore(self, s):
        """
        forward searching: f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g_fore[s] + self.h(s, self.s_goal)

    def f_value_back(self, s):
        """
        backward searching: f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g_back[s] + self.h(s, self.s_start)

    def h(self, s, goal):
        """
        Calculate heuristic value.
        :param s: current node (state)
        :param goal: goal node (state)
        :return: heuristic value
        """

        heuristic_type = self.heuristic_type

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False


class BestFirst(AStar):
    """BestFirst set the heuristics as the priority"""

    def searching(self):
        """
        Breadth-first Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN, (self.heuristic(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s

                    # best first set the heuristics as the priority
                    heapq.heappush(self.OPEN, (self.heuristic(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED


class Dijkstra(AStar):
    """Dijkstra set the cost as the priority"""

    def searching(self):
        """
        Breadth-first Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN, (0, self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s

                    # best first set the heuristics as the priority
                    heapq.heappush(self.OPEN, (new_cost, s_n))

        return self.extract_path(self.PARENT), self.CLOSED


class Plotting:
    def __init__(self, xI, xG):
        self.xI, self.xG = xI, xG
        self.env = ENV
        self.obs = self.env.obs_map()
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.ax.set_xlim(0, GLOBAL_CONFIG["width"])
        self.ax.set_ylim(0, GLOBAL_CONFIG["height"])
        self.ax.invert_yaxis()
        self.ax.set_aspect("equal")

    def update_obs(self, obs):
        self.obs = obs

    def animation(self, path, visited, name):
        self.plot_grid(name)
        self.plot_visited(visited)
        self.plot_path(path)
        plt.show()

    def animation_lrta(self, path, visited, name):
        self.plot_grid(name)
        cl = self.color_list_2()
        path_combine = []

        for k in range(len(path)):
            self.plot_visited(visited[k], cl[k])
            plt.pause(0.2)
            self.plot_path(path[k])
            path_combine += path[k]
            plt.pause(0.2)
        if self.xI in path_combine:
            path_combine.remove(self.xI)
        self.plot_path(path_combine)
        plt.show()

    def animation_ara_star(self, path, visited, name):
        self.plot_grid(name)
        cl_v, cl_p = self.color_list()

        for k in range(len(path)):
            self.plot_visited(visited[k], cl_v[k])
            self.plot_path(path[k], cl_p[k], True)
            plt.pause(0.5)

        plt.show()

    def animation_bi_astar(self, path, v_fore, v_back, name):
        self.plot_grid(name)
        self.plot_visited_bi(v_fore, v_back)
        self.plot_path(path)
        plt.show()

    def plot_grid(self, name):
        obs_x = [x[1] for x in self.obs]
        obs_y = [x[0] for x in self.obs]
        self.ax.scatter(obs_x, obs_y, c="black", s=1, zorder=1)
        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")
        # plt.plot(obs_x, obs_y, "sk")
        plt.title(name)

    def plot_visited(self, visited, cl="gray"):
        if self.xI in visited:
            visited.remove(self.xI)

        if self.xG in visited:
            visited.remove(self.xG)

        count = 0

        for x in visited:
            count += 1
            plt.plot(x[0], x[1], color=cl, marker="o")
            plt.gcf().canvas.mpl_connect("key_release_event", lambda event: [exit(0) if event.key == "escape" else None])

            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) * 2 / 3:
                length = 30
            else:
                length = 40
            #
            # length = 15

            if count % length == 0:
                plt.pause(0.001)
        plt.pause(0.01)

    def plot_path(self, path, cl="r", flag=False):
        path_x = [path[i][1] for i in range(len(path))]
        path_y = [path[i][0] for i in range(len(path))]

        if not flag:
            plt.plot(path_x, path_y, linewidth="1", color="r")
        else:
            plt.plot(path_x, path_y, linewidth="1", color=cl)

        plt.plot(self.xI[1], self.xI[0], "bs")
        plt.plot(self.xG[1], self.xG[0], "gs")

        plt.pause(0.01)

    def plot_visited_bi(self, v_fore, v_back):
        if self.xI in v_fore:
            v_fore.remove(self.xI)

        if self.xG in v_back:
            v_back.remove(self.xG)

        len_fore, len_back = len(v_fore), len(v_back)

        for k in range(max(len_fore, len_back)):
            if k < len_fore:
                plt.plot(v_fore[k][0], v_fore[k][1], linewidth="3", color="gray", marker="o")
            if k < len_back:
                plt.plot(v_back[k][0], v_back[k][1], linewidth="3", color="cornflowerblue", marker="o")

            plt.gcf().canvas.mpl_connect("key_release_event", lambda event: [exit(0) if event.key == "escape" else None])

            if k % 10 == 0:
                plt.pause(0.001)
        plt.pause(0.01)

    @staticmethod
    def color_list():
        cl_v = ["silver", "wheat", "lightskyblue", "royalblue", "slategray"]
        cl_p = ["gray", "orange", "deepskyblue", "red", "m"]
        return cl_v, cl_p

    @staticmethod
    def color_list_2():
        cl = [
            "silver",
            "steelblue",
            "dimgray",
            "cornflowerblue",
            "dodgerblue",
            "royalblue",
            "plum",
            "mediumslateblue",
            "mediumpurple",
            "blueviolet",
        ]
        return cl


if __name__ == "__main__":
    start = (149, 1604)
    goal = (1000, 71)
    plot = Plotting(start, goal)
    with utils.timer():
        astar = AStar(start, goal, "euclidean")
        path, visited = astar.searching()
    # print(path)
    # plot.plot_grid("a star")
    # plot.plot_path(path=path)
    # plt.pause(10000)

    # with utils.timer():
    #     bastar = BidirectionalAStar(start, goal, "euclidean")
    #     path, visited_fore, visited_back = bastar.searching()
    # plot.plot_grid("BidirectionalAStar")
    # plot.plot_path(path=path)
    # plt.pause(10000)

    # with utils.timer():
    #     bf = BestFirst(start, goal, "euclidean")
    #     path, visited = bf.searching()

    plot.plot_grid("BestFirst")
    plot.plot_path(path=path)
    plt.pause(10000)

    # path, visited = astar.searching_repeated_astar(2.5)
    # print(path)
    # plot.animation_ara_star(path, visited, "Repeated A*")
