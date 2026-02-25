"""网络流 IP 模型。

将路由问题建模为网络流整数规划：
- 决策变量：每条有向边 x_{u,v} ∈ {0, 1}
- 约束：流守恒（源+1，汇-1，中间0）
- 可选约束：转弯次数限制、容量约束
- 目标：最小化选用边的总代价

核心认知（来自 CLAUDE.md）：
- 路径问题本质 = 图上 0-1 决策
- 网络流在标准结构下 LP 松弛即得整数解（全单模性）
"""

from typing import Tuple, List, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum

Node = Tuple[int, ...]


class SolveStatus(Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    NOT_SOLVED = "not_solved"


@dataclass
class FlowSolution:
    """求解结果。

    Attributes:
        status: 求解状态
        path: 从源到汇的路径（节点列表）
        selected_edges: 被选中的边集合
        objective_value: 目标函数值
        turn_count: 转弯次数（如果设置了转弯约束）
    """
    status: SolveStatus = SolveStatus.NOT_SOLVED
    path: List[Node] = field(default_factory=list)
    selected_edges: Dict[Node, Node] = field(default_factory=dict)
    objective_value: float = float('inf')
    turn_count: int = 0


class FlowNetworkModel:
    """网络流 IP 模型。

    将体素网格上的最短路问题建模为网络流 IP，
    通过 OR-Tools SCIP 求解。

    支持：
    - 基本流守恒最短路
    - 转弯次数约束
    - 边代价权重（SDF 集成）
    - 容量约束

    Usage:
        model = FlowNetworkModel()
        model.set_graph(nodes, edges, costs)
        model.set_source_sink(start, goal)
        model.add_flow_conservation()
        model.set_max_turns(3)
        solution = model.solve()
    """

    def __init__(self):
        self._nodes: List[Node] = []
        self._node_set: Set[Node] = set()
        self._edges: List[Tuple[Node, Node]] = []
        self._edge_costs: Dict[Tuple[Node, Node], float] = {}
        self._source: Optional[Node] = None
        self._sink: Optional[Node] = None
        self._max_turns: Optional[int] = None
        self._directions: List[Tuple[int, ...]] = []

    def set_graph(
        self,
        nodes: List[Node],
        edges: List[Tuple[Node, Node]],
        costs: Optional[Dict[Tuple[Node, Node], float]] = None,
    ) -> None:
        """设置图结构。

        Args:
            nodes: 节点列表
            edges: 有向边列表 [(u, v), ...]
            costs: 边代价字典（默认每边代价 1.0）
        """
        self._nodes = list(nodes)
        self._node_set = set(nodes)
        self._edges = list(edges)
        if costs is not None:
            self._edge_costs = dict(costs)
        else:
            self._edge_costs = {e: 1.0 for e in edges}

    def set_source_sink(self, source: Node, sink: Node) -> None:
        """设置源点和汇点。"""
        self._source = source
        self._sink = sink

    def set_max_turns(self, max_turns: int) -> None:
        """设置最大转弯次数约束。"""
        self._max_turns = max_turns

    def set_directions(self, directions: List[Tuple[int, ...]]) -> None:
        """设置移动方向（用于转弯判定）。"""
        self._directions = directions

    def solve(self) -> FlowSolution:
        """构建并求解 IP 模型。

        Returns:
            FlowSolution 求解结果
        """
        from ortools.linear_solver import pywraplp

        if self._source is None or self._sink is None:
            raise ValueError("必须先设置源点和汇点")
        if not self._edges:
            raise ValueError("图中没有边")

        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            raise RuntimeError("无法创建 SCIP 求解器")

        # 1. 决策变量：每条有向边 x_{u,v} ∈ {0, 1}
        x = {}
        for u, v in self._edges:
            x[(u, v)] = solver.BoolVar(f"x_{u}_{v}")

        # 2. 流守恒约束
        for node in self._nodes:
            inflow = [x[e] for e in x if e[1] == node]
            outflow = [x[e] for e in x if e[0] == node]

            if node == self._source:
                solver.Add(sum(outflow) - sum(inflow) == 1)
            elif node == self._sink:
                solver.Add(sum(outflow) - sum(inflow) == -1)
            else:
                solver.Add(sum(outflow) - sum(inflow) == 0)
                # 防环：中间节点最多出度 1
                solver.Add(sum(outflow) <= 1)

        # 3. 转弯约束（可选）
        turn_vars = []
        if self._max_turns is not None:
            turn_vars = self._add_turn_constraints(solver, x)
            solver.Add(sum(turn_vars) <= self._max_turns)

        # 4. 目标函数：最小化加权边代价
        solver.Minimize(
            solver.Sum(
                self._edge_costs.get(e, 1.0) * x[e] for e in x
            )
        )

        # 5. 求解
        status = solver.Solve()

        solution = FlowSolution()

        if status == pywraplp.Solver.OPTIMAL:
            solution.status = SolveStatus.OPTIMAL
        elif status == pywraplp.Solver.FEASIBLE:
            solution.status = SolveStatus.FEASIBLE
        else:
            solution.status = SolveStatus.INFEASIBLE
            return solution

        # 6. 提取路径
        solution.objective_value = solver.Objective().Value()
        solution.selected_edges = {
            u: v for (u, v), var in x.items()
            if var.solution_value() > 0.5
        }

        # 串成路径
        path = [self._source]
        cur = self._source
        visited = set()
        while cur != self._sink and cur in solution.selected_edges:
            if cur in visited:
                break  # 防止无限循环
            visited.add(cur)
            cur = solution.selected_edges[cur]
            path.append(cur)
        solution.path = path

        if turn_vars:
            solution.turn_count = int(
                sum(t.solution_value() for t in turn_vars)
            )

        return solution

    def _add_turn_constraints(self, solver, x) -> list:
        """添加转弯约束变量和约束。

        对每个中间节点 v，枚举 (进入边, 离开边) 对，
        如果方向不一致则创建转弯指示变量。

        Returns:
            turn_vars 列表
        """
        turn_vars = []

        for v in self._nodes:
            if v == self._source or v == self._sink:
                continue

            in_edges = [e for e in x if e[1] == v]
            out_edges = [e for e in x if e[0] == v]

            for e_in in in_edges:
                for e_out in out_edges:
                    u = e_in[0]
                    w = e_out[1]

                    # 方向向量
                    vec1 = tuple(v[i] - u[i] for i in range(len(v)))
                    vec2 = tuple(w[i] - v[i] for i in range(len(v)))

                    if vec1 != vec2:
                        t = solver.BoolVar(f"turn_{u}_{v}_{w}")
                        # 线性化: t >= x_in + x_out - 1
                        solver.Add(t >= x[e_in] + x[e_out] - 1)
                        turn_vars.append(t)

        return turn_vars

    @classmethod
    def from_grid3d(
        cls,
        grid,
        source: Node,
        sink: Node,
        connectivity: int = 6,
        sdf_weight: float = 0.0,
    ) -> 'FlowNetworkModel':
        """从 Grid3D 构建网络流模型的工厂方法。

        Args:
            grid: Grid3D 实例
            source: 源点
            sink: 汇点
            connectivity: 6 或 26
            sdf_weight: SDF 代价权重

        Returns:
            配置好的 FlowNetworkModel
        """
        from ahl.geometry.voxel.grid3d import CellType, NEIGHBORS_6, NEIGHBORS_26
        import math

        offsets = NEIGHBORS_6 if connectivity == 6 else NEIGHBORS_26
        nx_size, ny_size, nz_size = grid.shape
        data = grid.data

        sdf = grid.get_sdf() if sdf_weight > 0 else None

        nodes = []
        edges = []
        costs = {}

        for i in range(nx_size):
            for j in range(ny_size):
                for k in range(nz_size):
                    if data[i, j, k] == CellType.OBSTACLE:
                        continue
                    node = (i, j, k)
                    nodes.append(node)

                    for off in offsets:
                        ni = i + int(off[0])
                        nj = j + int(off[1])
                        nk = k + int(off[2])
                        if (0 <= ni < nx_size and 0 <= nj < ny_size
                                and 0 <= nk < nz_size
                                and data[ni, nj, nk] != CellType.OBSTACLE):
                            neighbor = (ni, nj, nk)
                            edge = (node, neighbor)
                            edges.append(edge)

                            dist = math.sqrt(
                                (ni - i)**2 + (nj - j)**2 + (nk - k)**2
                            )
                            cost = dist
                            if sdf is not None and sdf_weight > 0:
                                cost += sdf_weight / (float(sdf[ni, nj, nk]) + 0.1)
                            costs[edge] = cost

        model = cls()
        model.set_graph(nodes, edges, costs)
        model.set_source_sink(source, sink)
        return model
