import time
import numpy as np
from astar import AStar, BestFirst, Dijkstra, BidirectionalAStar
from config import GLOBAL_CONFIG
from rrt import ALGORITHM_CONFIG, RRT, AlgorithmConfig, Node, has_collision
import utils


def test_path_algorithms(n=20):
    """测试路径规划算法的成功率、时间和路径长度"""
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
    # ALGORITHM_CONFIG = AlgorithmConfig(heuristic=True, bidirectional=True, adaptive_step=True, collision_method="bresenham")
    # 定义算法配置列表（名称，类，参数字典）
    configs = [
        ("AStar-euclidean", AStar, "euclidean"),
        ("AStar-manhattan", AStar, "manhattan"),
        ("BestFirst-euclidean", BestFirst, "euclidean"),
        ("BestFirst-manhattan", BestFirst, "manhattan"),
        ("BiAStar-euclidean", BidirectionalAStar, "euclidean"),
        ("BiAStar-manhattan", BidirectionalAStar, "manhattan"),
        ("improved RRT", RRT, ""),
        # ("Dijkstra", Dijkstra, "euclidean"),
    ]
    rrt = RRT(start=None, end=None)
    rrt.set_col_map(col_map)
    results = {}
    for config in configs:
        name, algo_class, params = config
        print(f"\n=== 正在测试算法：{name} ===")

        success = 0
        total_time = 0.0
        path_lengths = []

        for start, goal in test_cases:
            try:
                # 执行搜索并计时
                start_time = time.time()
                if algo_class == BidirectionalAStar:
                    path, _, _ = algo_class(start, goal, params).searching()
                elif algo_class == RRT:
                    start_node = Node(start[0], start[1])
                    end_node = Node(goal[0], goal[1])
                    rrt.set_start(start_node.row, start_node.col)
                    rrt.set_end(end_node.row, end_node.col)
                    rrt.search_path()
                    path = rrt.path_final
                else:
                    path, _ = algo_class(start, goal, params).searching()  # 假设返回(path, nodes_visited)
                elapsed = time.time() - start_time

                if path:  # 路径存在则记录成功
                    success += 1
                    total_time += elapsed
                    path_lengths.append(utils.calculate_path_len(path))
            except Exception as e:
                print(f"测试用例 {start}->{goal} 失败: {str(e)}")
                continue

        # 计算统计数据
        avg_time = total_time / success if success > 0 else 0
        avg_length = np.mean(path_lengths) if path_lengths else 0
        success_rate = success / len(test_cases)

        results[name] = {"success_rate": success_rate, "avg_time": avg_time, "avg_length": avg_length}

    # 打印结果表格
    print("\n=== 测试结果汇总 ===")
    print("{:<20} {:<15} {:<15} {:<15}".format("算法名称", "成功率", "平均时间(s)", "平均路径长度"))

    for algo_name, data in results.items():
        print("{:<20} {:<15.2%} {:<15.6f} {:<15.6f}".format(algo_name, data["success_rate"], data["avg_time"], data["avg_length"]))

    return results


if __name__ == "__main__":
    test_path_algorithms(n=100)
