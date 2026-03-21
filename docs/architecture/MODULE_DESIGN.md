# AHL 项目模块设计规划

## 总体架构

```
src/ahl/
├── geometry/          # 几何引擎层 - CAD/Mesh输入与体素化
├── routing/           # 核心路由层 - 路径规划与网络构建
├── optimization/      # 局部微调层 - IP规划与约束优化
├── blender/           # Blender集成 - CAD模型处理
├── api/               # Web API - FastAPI服务
├── ui/                # 桌面UI - PyQt6界面（3D主程序）
├── grid2d/            # 2D网格编辑器（已存在，保留用于测试验证）
├── utils/             # 通用工具
├── config/            # 配置管理
└── main.py            # 命令行入口
```

---

## 一、几何引擎层 (`geometry/`)

**职责**: 处理3D几何模型输入，生成体素网格与SDF

### 模块结构
```
geometry/
├── __init__.py
├── voxel/
│   ├── __init__.py
│   ├── grid3d.py          # 三维体素网格核心类
│   ├── voxelizer.py       # Mesh → Voxel转换
│   └── sdf.py             # SDF计算与梯度
├── mesh/
│   ├── __init__.py
│   ├── loader.py          # STL/OBJ/FBX加载器
│   ├── mesh_ops.py        # Mesh基础操作（法向量、面积等）
│   └── collision.py       # 碰撞检测（Ray-casting）
└── spatial/
    ├── __init__.py
    ├── kdtree.py          # KD-Tree索引（端点聚类用）
    └── octree.py          # 八叉树（稀疏体素加速）
```

### 核心类设计

#### `Grid3D` - 三维体素网格
```python
class Grid3D:
    """三维体素网格，支持SDF与路径规划"""
    data: np.ndarray        # (Nx, Ny, Nz) - 体素状态
    sdf: np.ndarray         # (Nx, Ny, Nz) - 有符号距离场
    resolution: float       # 体素尺寸（mm）
    origin: np.ndarray      # 世界坐标原点

    def get_neighbors(pos, connectivity=26) -> List[Tuple]
    def is_valid(pos) -> bool
    def world_to_voxel(coords) -> np.ndarray
    def voxel_to_world(indices) -> np.ndarray
```

#### `SDFComputer` - SDF计算器
```python
class SDFComputer:
    """基于Mesh或体素的SDF快速计算"""
    def compute_from_voxel(grid: Grid3D) -> np.ndarray
    def compute_from_mesh(mesh: Mesh) -> np.ndarray
    def gradient(sdf: np.ndarray) -> np.ndarray  # ∇SDF
```

---

## 二、核心路由层 (`routing/`)

**职责**: 单路径搜索、多终点网络构建、Steiner Tree近似

### 模块结构
```
routing/
├── __init__.py
├── single/
│   ├── __init__.py
│   ├── astar.py           # A*搜索（SDF引导）
│   ├── dijkstra.py        # Dijkstra算法
│   └── cost_function.py   # 代价函数（SDF惩罚项）
├── network/
│   ├── __init__.py
│   ├── steiner_tree.py    # Steiner Tree近似算法
│   ├── backbone.py        # 主干网络生成
│   ├── branch.py          # 支线连接生成
│   └── clustering.py      # 终端聚类（K-means）
└── graph/
    ├── __init__.py
    ├── graph_builder.py   # 体素网格→图结构
    └── path_ops.py        # 路径平滑/合并/简化
```

### 核心算法

#### `AStarSDF` - SDF引导的A*
```python
class AStarSDF:
    """A*搜索，代价函数包含SDF惩罚项避免贴边"""
    def __init__(self, grid: Grid3D, w_dist=1.0, w_sdf=0.5):
        self.grid = grid
        self.w_dist = w_dist      # 距离权重
        self.w_sdf = w_sdf        # SDF惩罚权重

    def search(start, goal) -> Optional[Path]
    def cost(node, next_node) -> float
```

#### `BackboneBuilder` - 主干网络构建
```python
class BackboneBuilder:
    """先构建主干，再生成支线"""
    def build(terminals: List[Point]) -> Backbone
    def cluster_terminals(terminals, n_clusters) -> List[Cluster]
    def connect_clusters(clusters) -> Graph
```

#### `BranchRouter` - 支线路由（贴主干）
```python
class BranchRouter:
    """通过代价偏置使支线优先复用主干"""
    def __init__(self, backbone: Backbone, cost_bias=0.3):
        self.backbone = backbone
        self.cost_bias = cost_bias  # 主干边代价折扣

    def route(terminal: Point, backbone_node: Point) -> Path
```

---

## 三、局部微调层 (`optimization/`)

**职责**: IP规划、约束优化、安全距离强化

### 模块结构
```
optimization/
├── __init__.py
├── ip/
│   ├── __init__.py
│   ├── flow_network.py    # 网络流建模
│   ├── variables.py       # 决策变量（边选用0-1）
│   └── constraints.py     # 流守恒/容量/路径约束
├── ortools/
│   ├── __init__.py
│   ├── solver.py          # OR-Tools CP-SAT求解器封装
│   └── model_builder.py   # 约束模型构建
└── refinement/
    ├── __init__.py
    ├── local_search.py    # 局部搜索优化
    └── smoothing.py       # 路径平滑（B样条）
```

### 核心类

#### `FlowNetworkModel` - 网络流IP模型
```python
class FlowNetworkModel:
    """将路由问题建模为网络流IP"""
    def add_edge_variable(u, v) -> Variable
    def add_flow_conservation(node) -> Constraint
    def add_capacity_constraint(edge, capacity) -> Constraint
    def solve() -> Solution
```

---

## 四、Blender集成 (`blender/`)

**职责**: 使用Blender Python API处理CAD模型

### 模块结构
```
blender/
├── __init__.py
├── importer.py            # 导入CAD模型（STEP/IGES）
├── mesh_processor.py      # Mesh清理/修复/简化
├── voxelizer.py           # Blender → Voxel转换（利用bmesh）
└── exporter.py            # 导出结果（布线Curve对象）
```

**注意**: 此模块依赖Blender环境，需要在Blender内嵌Python中运行

---

## 五、Web API (`api/`)

**职责**: 提供RESTful API服务

### 模块结构
```
api/
├── __init__.py
├── app.py                 # FastAPI应用实例
├── routes/
│   ├── __init__.py
│   ├── routing.py         # 路由计算端点
│   ├── geometry.py        # 几何处理端点
│   └── health.py          # 健康检查
├── schemas/
│   ├── __init__.py
│   ├── request.py         # 请求Schema（Pydantic）
│   └── response.py        # 响应Schema
└── middleware/
    ├── __init__.py
    └── error_handler.py   # 统一错误处理
```

---

## 六、桌面UI (`ui/`)

**职责**: PyQt6主界面（3D可视化与交互）

### 模块结构
```
ui/
├── __init__.py
├── main_window.py         # 主窗口
├── widgets/
│   ├── __init__.py
│   ├── voxel_viewer.py    # 3D体素可视化（OpenGL）
│   ├── path_viewer.py     # 路径显示
│   ├── terminal_editor.py # 接线终端编辑器
│   └── property_panel.py  # 属性面板
├── dialogs/
│   ├── __init__.py
│   ├── import_dialog.py   # 导入模型对话框
│   └── settings_dialog.py # 设置对话框
└── toolbar.py             # 工具栏
```

---

## 七、通用工具 (`utils/`)

```
utils/
├── __init__.py
├── logger.py              # 日志配置
├── io.py                  # 文件IO（npz/json）
├── validators.py          # 输入验证
└── math_utils.py          # 数学工具（向量/矩阵）
```

---

## 八、配置管理 (`config/`)

```
config/
├── __init__.py
├── settings.py            # 全局设置（Pydantic Settings）
└── defaults.py            # 默认参数
```

---

## 实施顺序建议

### Phase 1: 基础设施（当前）
1. `geometry/voxel/grid3d.py` - 三维网格核心
2. `geometry/voxel/sdf.py` - SDF计算
3. `routing/single/astar.py` - A*基础实现
4. `utils/` - 工具函数

### Phase 2: 路由核心
1. `routing/single/cost_function.py` - 代价函数完善
2. `routing/graph/` - 图构建与操作
3. `routing/network/clustering.py` - 终端聚类
4. `routing/network/backbone.py` - 主干生成

### Phase 3: 工程化
1. `blender/` - CAD集成
2. `optimization/` - IP规划
3. `api/` - Web服务
4. `ui/` - 3D界面

---

## 与现有grid2d的关系

`grid2d/` 保留作为：
- 2D算法原型验证
- 单元测试基准
- 教学演示工具

3D系统在 `geometry/`, `routing/`, `optimization/` 中独立实现，不依赖grid2d。

---

## 技术债务管理

### 避免过度抽象
- 不提前设计接口层次
- 先实现具体算法，稳定后再抽象
- 优先numpy数组，而非复杂对象层次

### 模块隔离
- 每层可独立测试
- 避免跨层直接调用（geometry ↛ optimization）
- 通过数据传递解耦（numpy数组/Graph对象）

---

## 数据流示意

```
CAD Model (STEP/STL)
    ↓ blender/importer.py
Mesh (vertices, faces)
    ↓ geometry/voxel/voxelizer.py
Grid3D (voxel array)
    ↓ geometry/voxel/sdf.py
Grid3D + SDF
    ↓ routing/graph/graph_builder.py
Graph (nodes, edges, weights)
    ↓ routing/network/backbone.py
Backbone Network
    ↓ routing/network/branch.py
Full Routing Solution
    ↓ optimization/refinement/
Optimized Path
    ↓ blender/exporter.py
Blender Curve Objects
```

---

## 依赖关系图

```
[blender] ──────→ [geometry] ──────→ [routing] ──────→ [optimization]
                      ↓                  ↓                    ↓
                  [utils]            [graph]             [ortools]
                                         ↓
                                     [api/ui]
```

---

## 测试策略

```
test/
├── geometry/
│   ├── test_grid3d.py
│   ├── test_sdf.py
│   └── test_voxelizer.py
├── routing/
│   ├── test_astar.py
│   ├── test_steiner.py
│   └── test_backbone.py
├── optimization/
│   └── test_flow_network.py
└── integration/
    └── test_full_pipeline.py
```

每个模块需要：
1. 单元测试（覆盖率 > 80%）
2. 集成测试（关键路径）
3. 性能基准测试（大规模数据）

---

**最后更新**: 2026-02-07
