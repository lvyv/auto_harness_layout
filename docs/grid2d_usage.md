# Grid2D 编辑器使用指南

## 简介

Grid2D 是一个完整的二维网格编辑器和 A* 路径规划系统，支持：
- 最大 1000×1000 的二维网格
- PyQt6 可视化界面，支持交互式编辑
- 支持设置多个起点和多个终点
- A* 算法，带 SDF（符号距离场）惩罚项
- 保存/加载功能（.npz 格式）

## 快速启动

### 1. 启动图形界面

```bash
# 方式 1: 使用 uv 运行模块
uv run python -m ahl.grid2d

# 方式 2: 使用启动脚本
uv run python run_grid2d_editor.py
```

### 2. 界面操作

#### 工具栏
- **New**: 创建新网格
- **Save**: 保存网格到 .npz 文件
- **Load**: 从 .npz 文件加载网格
- **Reset View**: 重置视图以适应网格大小
- **Edit Mode**: 选择编辑模式
  - Draw Obstacle: 绘制障碍物（黑色）
  - Place Start: 放置起点（绿色）
  - Place End: 放置终点（红色）
  - Eraser: 橡皮擦工具

#### 网格视图操作
- **左键点击**: 编辑单元格（根据当前编辑模式）
- **左键拖拽**: 连续编辑
- **右键拖拽**: 平移视图
- **鼠标滚轮**: 缩放视图

#### 控制面板
- **Grid Info**: 显示网格尺寸和起点/终点数量
- **A* Configuration**:
  - Allow Diagonal Movement: 是否允许对角移动
  - SDF Weight: SDF 惩罚权重（0-10，值越大路径离障碍物越远）
  - Max Iterations: 最大迭代次数（防止死循环）
- **Points**: 显示所有起点和终点坐标
- **Run A* Pathfinding**: 运行 A* 算法查找所有起点到终点的路径
- **Clear All Paths**: 清除所有已计算的路径

## 使用示例

### 示例 1: 简单路径规划

1. 启动编辑器
2. 选择 "Place Start" 模式，在网格上点击设置起点（绿色）
3. 选择 "Place End" 模式，在网格上点击设置终点（红色）
4. 点击 "Run A* Pathfinding" 按钮
5. 路径将以蓝色显示

### 示例 2: 绕障碍物路径

1. 选择 "Draw Obstacle" 模式
2. 在网格上拖拽鼠标绘制障碍物墙（黑色）
3. 设置起点和终点
4. 调整 SDF Weight（例如设置为 1.0）
5. 运行 A* 路径规划
6. 观察路径如何绕过障碍物，并保持一定距离

### 示例 3: 多起点多终点

1. 使用 "Place Start" 模式放置多个起点
2. 使用 "Place End" 模式放置多个终点
3. 运行 A* 路径规划
4. 系统会计算所有起点到所有终点的路径（N×M 条路径）

### 示例 4: 对角移动对比

1. 设置起点和终点
2. 不勾选 "Allow Diagonal Movement"，运行 A*
3. 观察路径长度（曼哈顿距离）
4. 清除路径，勾选 "Allow Diagonal Movement"，再次运行
5. 观察路径长度减少（欧氏距离）

## 编程接口

### 基本使用

```python
from ahl.grid2d import Grid, CellType, astar_search, AStarConfig

# 创建网格
grid = Grid(width=50, height=50)

# 添加障碍物
grid.fill_rect(10, 10, 20, 20, CellType.OBSTACLE)

# 设置起点和终点
start_idx = grid.add_start(5, 5)
end_idx = grid.add_end(45, 45)

# 配置 A* 参数
config = AStarConfig(
    diagonal_move=True,
    sdf_weight=0.5,
    max_iterations=1_000_000
)

# 运行 A* 路径规划
path = astar_search(grid, (5, 5), (45, 45), config)

if path:
    print(f"找到路径，长度: {len(path)}")
    print(f"路径: {path}")
else:
    print("未找到路径")
```

### 保存和加载

```python
from ahl.grid2d.io import save_grid, load_grid

# 保存网格
save_grid(grid, "my_grid.npz")

# 加载网格
loaded_grid = load_grid("my_grid.npz")
```

### 批量路径规划

```python
from ahl.grid2d.core.astar import batch_astar

# 计算多个起点到多个终点的所有路径
starts = [(0, 0), (5, 5), (10, 10)]
goals = [(40, 40), (45, 45)]

results = batch_astar(grid, starts, goals, config)

for (start, goal), path in results.items():
    if path:
        print(f"路径 {start} -> {goal}: 长度 {len(path)}")
    else:
        print(f"路径 {start} -> {goal}: 无解")
```

## 技术细节

### A* 算法与 SDF 惩罚

本系统的 A* 实现遵循 CLAUDE.md 中的工程原则："在搜索阶段编码偏好，而非后处理平滑"。

代价函数：
```
cost = movement_cost + sdf_penalty
sdf_penalty = sdf_weight / (sdf_value + epsilon)
```

其中：
- `movement_cost`: 移动代价（4-connected: 1.0, diagonal: 1.414）
- `sdf_value`: 到最近障碍物的距离
- `sdf_weight`: 权重参数（可调节）
- `epsilon`: 防止除零（默认 0.1）

**效果**：距离障碍物越近，惩罚越大，路径自然远离障碍物。

### SDF（符号距离场）计算

使用 `scipy.ndimage.distance_transform_edt` 计算欧氏距离变换：
- 输入：网格障碍物 mask
- 输出：每个点到最近障碍物的距离
- 缓存机制：仅在障碍物改变时重新计算

### 文件格式（NPZ）

NPZ 文件包含：
- `cells`: (H, W) int8 网格数组
- `starts`: (N, 2) 起点坐标
- `ends`: (M, 2) 终点坐标
- `path_<start_idx>_<end_idx>`: 每条路径的坐标数组
- `config_*`: 网格配置参数
- `metadata_*`: 元数据（版本、时间戳）

压缩存储，1000×1000 网格文件通常 < 5KB（若大部分为空）。

## 性能

### 测试性能
- 创建 1000×1000 网格：< 1 秒
- 500×500 网格 A* 路径规划：< 5 秒
- 视口裁剪：仅渲染可见区域，支持流畅缩放

### 优化建议
- 对于大网格（> 500×500），建议使用较低的 SDF 权重（< 1.0）
- 对于复杂障碍物环境，可以增加 max_iterations
- 批量路径规划时，可考虑并行计算（未实现）

## 故障排除

### 问题：路径贴着障碍物边缘
**解决**：增加 SDF Weight（例如从 0.5 提高到 1.5）

### 问题：找不到路径
**原因**：
1. 起点或终点被障碍物包围
2. 障碍物完全阻断路径
3. Max Iterations 设置过低

**解决**：检查网格设置，清除多余障碍物，或增加迭代次数

### 问题：路径不是最短路径
**说明**：这是正常的！由于 SDF 惩罚项，路径会选择"更安全"的路线而非"最短"路线。这是工程上的权衡。如果需要最短路径，将 SDF Weight 设为 0。

### 问题：大网格运行缓慢
**解决**：
1. 使用视口裁剪（自动启用）
2. 减少同时计算的路径数量
3. 降低网格分辨率

## 扩展方向

根据 CLAUDE.md 中的规划，未来可扩展功能：
1. 路径复用可视化（Steiner Tree 准备）
2. 3D 扩展（Voxel Grid）
3. 批量路径规划优化
4. 交互式 A*（实时显示搜索过程）
5. SDF 热力图可视化
6. 路径统计（长度、代价、安全距离）
7. CAD 导入（DXF/SVG）

## 测试

运行完整测试套件：

```bash
# 运行所有 grid2d 测试
uv run pytest test/grid2d/ -v

# 运行特定测试
uv run pytest test/grid2d/test_astar.py -v
uv run pytest test/grid2d/test_grid.py -v
uv run pytest test/grid2d/test_sdf.py -v
uv run pytest test/grid2d/test_io.py -v
```

测试覆盖：
- 41 个测试用例
- 覆盖核心功能、边界情况、性能测试
- 100% 通过率

## 贡献者

- 基于 CLAUDE.md 中的工程原则设计
- 遵循"可解释性 > 稳定性 > 工程可控性"的优先级
- 这是工程系统，不是论文实现
