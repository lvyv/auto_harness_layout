"""
Grid2D 使用示例

演示如何使用 Grid2D 模块进行路径规划
"""

from ahl.grid2d import Grid, CellType, astar_search, AStarConfig
from ahl.grid2d.io import save_grid, load_grid


def example_1_simple_path():
    """示例 1: 简单直线路径"""
    print("=" * 60)
    print("示例 1: 简单直线路径")
    print("=" * 60)

    grid = Grid(20, 20)
    start = (5, 5)
    goal = (5, 15)

    path = astar_search(grid, start, goal)

    print(f"起点: {start}")
    print(f"终点: {goal}")
    print(f"路径长度: {len(path)}")
    print(f"路径: {path}")
    print()


def example_2_obstacle_avoidance():
    """示例 2: 绕障碍物路径"""
    print("=" * 60)
    print("示例 2: 绕障碍物路径")
    print("=" * 60)

    grid = Grid(30, 30)

    # 创建垂直墙
    for r in range(5, 25):
        grid.set_cell(r, 15, CellType.OBSTACLE)

    start = (15, 5)
    goal = (15, 25)

    # 不带 SDF 惩罚
    config_no_sdf = AStarConfig(sdf_weight=0.0)
    path_no_sdf = astar_search(grid, start, goal, config_no_sdf)

    # 带 SDF 惩罚
    config_with_sdf = AStarConfig(sdf_weight=1.0)
    path_with_sdf = astar_search(grid, start, goal, config_with_sdf)

    print(f"起点: {start}, 终点: {goal}")
    print(f"障碍物: 垂直墙从 (5, 15) 到 (24, 15)")
    print(f"路径长度（无 SDF）: {len(path_no_sdf)}")
    print(f"路径长度（有 SDF）: {len(path_with_sdf)}")
    print(f"提示: SDF 惩罚使路径远离障碍物，可能导致路径略长但更安全")
    print()


def example_3_diagonal_movement():
    """示例 3: 对角移动对比"""
    print("=" * 60)
    print("示例 3: 对角移动对比")
    print("=" * 60)

    grid = Grid(20, 20)
    start = (0, 0)
    goal = (10, 10)

    # 不允许对角移动
    config_no_diag = AStarConfig(diagonal_move=False)
    path_no_diag = astar_search(grid, start, goal, config_no_diag)

    # 允许对角移动
    config_diag = AStarConfig(diagonal_move=True)
    path_diag = astar_search(grid, start, goal, config_diag)

    print(f"起点: {start}, 终点: {goal}")
    print(f"路径长度（仅 4-connected）: {len(path_no_diag)}")
    print(f"路径长度（8-connected）: {len(path_diag)}")
    print(f"缩短比例: {(1 - len(path_diag)/len(path_no_diag))*100:.1f}%")
    print()


def example_4_multiple_paths():
    """示例 4: 多起点多终点"""
    print("=" * 60)
    print("示例 4: 多起点多终点")
    print("=" * 60)

    from ahl.grid2d.core.astar import batch_astar

    grid = Grid(50, 50)

    # 添加一些障碍物
    grid.fill_rect(20, 20, 30, 30, CellType.OBSTACLE)

    starts = [(5, 5), (5, 45), (45, 5)]
    goals = [(45, 45), (25, 5)]

    config = AStarConfig(diagonal_move=True, sdf_weight=0.5)
    results = batch_astar(grid, starts, goals, config)

    print(f"起点: {starts}")
    print(f"终点: {goals}")
    print(f"总路径数: {len(starts)} × {len(goals)} = {len(results)}")
    print()

    for (start, goal), path in results.items():
        if path:
            print(f"  {start} → {goal}: 长度 {len(path)}")
        else:
            print(f"  {start} → {goal}: 无解")
    print()


def example_5_save_load():
    """示例 5: 保存和加载网格"""
    print("=" * 60)
    print("示例 5: 保存和加载网格")
    print("=" * 60)

    # 创建网格
    grid = Grid(30, 30)

    # 添加障碍物
    grid.fill_rect(10, 10, 20, 20, CellType.OBSTACLE)

    # 添加起点和终点
    grid.add_start(5, 5)
    grid.add_start(25, 25)
    grid.add_end(5, 25)
    grid.add_end(25, 5)

    # 保存
    filename = "example_grid.npz"
    save_grid(grid, filename)
    print(f"网格已保存到: {filename}")

    # 加载
    loaded_grid = load_grid(filename)
    print(f"网格已加载: {loaded_grid.width}×{loaded_grid.height}")
    print(f"起点数: {len(loaded_grid.starts)}")
    print(f"终点数: {len(loaded_grid.ends)}")
    print(f"起点: {loaded_grid.starts}")
    print(f"终点: {loaded_grid.ends}")
    print()

    # 清理
    import os
    os.remove(filename)
    print(f"示例文件已删除")
    print()


def example_6_sdf_effect():
    """示例 6: SDF 权重对路径的影响"""
    print("=" * 60)
    print("示例 6: SDF 权重对路径的影响")
    print("=" * 60)

    grid = Grid(40, 40)

    # 创建走廊（两侧有障碍物）
    for r in range(10, 30):
        grid.set_cell(r, 15, CellType.OBSTACLE)
        grid.set_cell(r, 25, CellType.OBSTACLE)

    start = (20, 0)
    goal = (20, 39)

    weights = [0.0, 0.5, 1.0, 2.0]

    print(f"场景: 走廊（两侧障碍物），从 {start} 到 {goal}")
    print(f"测试不同 SDF 权重的效果:")
    print()

    for weight in weights:
        config = AStarConfig(sdf_weight=weight)
        path = astar_search(grid, start, goal, config)

        # 计算路径中心度（距离中心线的平均距离）
        center_col = 20
        avg_distance = sum(abs(col - center_col) for _, col in path) / len(path)

        print(f"  SDF Weight = {weight:.1f}: 路径长度 = {len(path)}, 平均偏离中心 = {avg_distance:.2f}")

    print()
    print("提示: SDF 权重越高，路径越倾向于走廊中心")
    print()


def main():
    """运行所有示例"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + " " * 15 + "Grid2D 使用示例" + " " * 27 + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")

    example_1_simple_path()
    example_2_obstacle_avoidance()
    example_3_diagonal_movement()
    example_4_multiple_paths()
    example_5_save_load()
    example_6_sdf_effect()

    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
    print()
    print("提示: 运行 'uv run python -m ahl.grid2d' 启动图形界面")
    print()


if __name__ == '__main__':
    main()
