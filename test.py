import numpy as np
from config import GLOBAL_CONFIG
from rrt import ALGORITHM_CONFIG, AlgorithmConfig, Node, has_collision
import utils


def test_all_with_config(n=20, configs=None):
    global ALGORITHM_CONFIG

    import cProfile
    import os
    import glob

    np.random.seed(999)
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
    ALGORITHM_CONFIG = AlgorithmConfig(heuristic=True, bidirectional=True, adaptive_step=True, collision_method="bresenham")

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
            ("Bidirectional", AlgorithmConfig(heuristic=False, bidirectional=True, adaptive_step=False, collision_method="discrete")),
            ("+Bresenham", AlgorithmConfig(heuristic=False, bidirectional=True, adaptive_step=False, collision_method="bresenham")),
            ("+Heuristic", AlgorithmConfig(heuristic=True, bidirectional=True, adaptive_step=False, collision_method="bresenham")),
            ("+AdaptiveStep", AlgorithmConfig(heuristic=False, bidirectional=True, adaptive_step=True, collision_method="bresenham")),
            ("All",),
        ]

    # 生成测试用例（所有配置共享同一组测试用例）

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
