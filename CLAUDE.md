# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 项目定位

**工程级三维自动布线系统 (Auto Harness Layout)**

核心场景：
- 汽车线束/管路自动走线
- 复杂CAD/Mesh几何环境下的路径生成
- 多终点、多约束的连接网络构建

目标不是学术最优解，而是：
> **在大规模、复杂几何、强工程约束下，生成稳定、可解释、可扩展的可用方案**

---

## 技术栈与开发环境

### 核心依赖
- **Python**: 3.10-3.13
- **Package Manager**: uv（已配置清华镜像）
- **Web Framework**: FastAPI + Uvicorn
- **UI**: PyQt6
- **Scientific**: numpy, matplotlib, pyproj
- **3D Geometry**: Blender 4.x (Python API)

### 开发命令
```bash
# 环境设置
uv sync
source .venv/Scripts/activate  # Windows Git Bash

# 运行
ahl                    # 命令行入口
python -m ahl.main     # 模块方式

# 测试
python -m pytest test/
```

### 项目结构
```
src/ahl/        - 主包（当前处于早期开发阶段）
test/           - 测试
docs/
  reference/    - 学术论文（3D medial axis, Steiner tree, voxel routing等）
  memo/         - 算法备忘录
  tasks/        - 任务追踪
```

---

## 核心技术架构

### 1. 几何建模
- **输入**: CAD/STL/Mesh模型（不规则曲面、薄壁零件）
- **空间表示**: 三维体素网格（Voxel Grid）
  - 由Mesh体素化或SDF生成
  - 状态标记: `free` / `obstacle` / `surface`

### 2. 问题建模
- **单路径**: A*/Dijkstra + SDF引导搜索
- **多终点网络**: Steiner Tree Problem（NP-hard）
  - 工程重点：路径复用、交叉控制、规则可控

### 3. 距离场（SDF）
- 作用：障碍物最小距离表示
- 用途：碰撞检测、安全距离约束、路径优化
- 性质：|∇SDF| ≈ 1，可作为启发式引导

---

## 工程决策与原则

### 整数规划（IP）视角
- 路径问题本质 = 图上0-1决策（边是否选用）
- 网络流在标准结构下LP松弛即得整数解（全单模性）
- 工业系统"假装不是IP，但行为等价于IP"

### Steiner Tree工程化
- **分层建模**: 主干（Trunk）+ 支线（Branch）
- **执行顺序**: 先构建稳定主干，再生成支线
- **复用机制**: 通过代价函数偏置（非硬约束）
  - 主干边降低traversal cost
  - 远离主干增加惩罚项

### A*搜索优化
- **问题**: 倾向贴障碍物边缘
- **解决**: cost函数加入SDF惩罚项（距离越近代价越高）
- **原则**: 在搜索阶段编码偏好，而非后处理平滑

### 工具约束
- **Blender**: 使用原生API（不依赖trimesh等外部库）
- **数据格式**: numpy数组，npy/npz中间格式
- **取舍**: 稳定性 > 精度

---

## 总体原则

优先级排序：
1. **可解释性** - 工程师可理解、可调试
2. **稳定性** - 在各种输入下鲁棒
3. **工程可控性** - 支持人工干预和参数调节

不追求：
- 纯理论最优解
- 复杂但不可维护的模型

> **这是工程系统，不是论文实现**

---

## 近期关键结论（动态更新区）

> 本节固化最近2-3天内反复讨论、具有长期工程价值的决策。

### 1. 多终点布线必然走向IP
- 问题：路径复用、共享、主干 = 离散全局决策
- 结论：无法用独立最短路径解决，本质是Network Design/IP结构

### 2. 主干网络先行
- 共识：不一次性对所有端点做全局STP
- 做法：先生成Backbone，再分组/逐个生成支线
- 优势：主干稳定可解释，支线容忍局部次优

### 3. 支线"贴主干"用代价而非规则
- 错误做法：硬规则"必须走主干"
- 正确做法：代价函数偏置（降低主干cost或增加远离惩罚）
- 本质：对IP目标函数的工程化近似

### 4. 最短路径 ⇄ IP等价性
- 认知：最短路可建模为IP（决策变量=边选用，约束=流守恒）
- 原因：网络矩阵全单模性（Totally Unimodular）
- 启示：大量工业寻路系统结果等价于IP但不显式调用求解器

---

## Claude使用说明

### 文件维护协议
当用户说"记住这个"、"加入项目背景"、"这是重要设计决策"时：
1. 提取工程级抽象结论
2. 以精炼风格更新本文件相应章节
3. 近期结论放入"动态更新区"，稳定后迁移到对应架构章节

### 避免写入
- 临时调试信息
- 冗长对话记录
- 过于具体的代码片段（除非是核心算法模式）

---

## 参考资源

`docs/reference/` 中的学术论文涵盖：
- 3D medial axis提取算法
- 体素建模与膨胀算子
- 聚类与KDTree索引
- 约束优化框架（OR-Tools相关）
