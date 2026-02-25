"""OR-Tools 求解器统一封装。

提供对 OR-Tools 线性求解器 (SCIP/GLOP) 和 CP-SAT 的统一接口。
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum

from ortools.linear_solver import pywraplp


class SolverType(Enum):
    SCIP = "SCIP"       # 混合整数规划
    GLOP = "GLOP"       # 纯线性规划
    CBC = "CBC"         # 开源 MIP


@dataclass
class SolverResult:
    """求解结果。"""
    optimal: bool
    feasible: bool
    objective: float
    wall_time_ms: float


class ORToolsSolver:
    """OR-Tools 求解器封装。

    提供创建求解器、设置参数、检查求解状态的统一接口。

    Usage:
        solver = ORToolsSolver.create(SolverType.SCIP)
        x = solver.BoolVar("x")
        solver.Add(x == 1)
        solver.Minimize(x)
        result = ORToolsSolver.solve(solver)
    """

    @staticmethod
    def create(
        solver_type: SolverType = SolverType.SCIP,
        time_limit_ms: Optional[int] = None,
    ) -> pywraplp.Solver:
        """创建求解器实例。

        Args:
            solver_type: 求解器类型
            time_limit_ms: 时间限制（毫秒）

        Returns:
            pywraplp.Solver 实例

        Raises:
            RuntimeError: 求解器不可用
        """
        solver = pywraplp.Solver.CreateSolver(solver_type.value)
        if solver is None:
            raise RuntimeError(f"无法创建 {solver_type.value} 求解器")

        if time_limit_ms is not None:
            solver.SetTimeLimit(time_limit_ms)

        return solver

    @staticmethod
    def solve(solver: pywraplp.Solver) -> SolverResult:
        """执行求解并返回结果。

        Args:
            solver: 配置好的求解器

        Returns:
            SolverResult
        """
        status = solver.Solve()

        return SolverResult(
            optimal=(status == pywraplp.Solver.OPTIMAL),
            feasible=(status in (
                pywraplp.Solver.OPTIMAL,
                pywraplp.Solver.FEASIBLE,
            )),
            objective=solver.Objective().Value() if status in (
                pywraplp.Solver.OPTIMAL,
                pywraplp.Solver.FEASIBLE,
            ) else float('inf'),
            wall_time_ms=solver.WallTime(),
        )

    @staticmethod
    def is_available(solver_type: SolverType = SolverType.SCIP) -> bool:
        """检查求解器是否可用。"""
        solver = pywraplp.Solver.CreateSolver(solver_type.value)
        return solver is not None
