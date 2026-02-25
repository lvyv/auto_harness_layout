"""路由计算端点。"""

from fastapi import APIRouter, HTTPException

from ahl.api.schemas.request import RoutingRequest, BackboneRequest
from ahl.api.schemas.response import RoutingResponse, PathResult, BackboneResponse
from ahl.api.routes.geometry import get_grid
from ahl.routing.single.astar import AStarSDF
from ahl.routing.graph.path_ops import path_length, count_turns

router = APIRouter(prefix="/routing", tags=["routing"])


@router.post("/path", response_model=RoutingResponse)
async def compute_path(req: RoutingRequest):
    """计算单条路径。

    支持 A* 和 IP 两种求解方式。
    """
    grid = get_grid()
    start = req.start.as_tuple()
    goal = req.goal.as_tuple()

    try:
        if req.use_ip:
            return _solve_ip(grid, start, goal, req)
        else:
            return _solve_astar(grid, start, goal, req)
    except ValueError as e:
        return RoutingResponse(success=False, error=str(e))
    except Exception as e:
        return RoutingResponse(success=False, error=f"内部错误: {e}")


@router.post("/backbone", response_model=BackboneResponse)
async def compute_backbone(req: BackboneRequest):
    """构建主干网络并路由支线。"""
    grid = get_grid()
    terminals = [p.as_tuple() for p in req.terminals]

    try:
        from ahl.routing.network.backbone import BackboneBuilder, BranchRouter

        builder = BackboneBuilder(grid, w_sdf=req.w_sdf)
        backbone = builder.build(
            terminals,
            n_clusters=req.n_clusters,
        )

        router_obj = BranchRouter(
            grid, backbone,
            cost_bias=req.cost_bias,
            w_sdf=req.w_sdf,
        )
        solution = router_obj.route_all(terminals)

        # 构建响应
        backbone_paths = []
        for key, path in backbone.paths.items():
            backbone_paths.append(PathResult(
                path=[list(p) for p in path],
                length=path_length(path),
                turns=count_turns(path),
                method="backbone_astar",
            ))

        branch_paths = []
        for terminal, path in solution.branches.items():
            if len(path) > 1:
                branch_paths.append(PathResult(
                    path=[list(p) for p in path],
                    length=path_length(path),
                    turns=count_turns(path),
                    method="branch_astar",
                ))

        return BackboneResponse(
            success=True,
            backbone_paths=backbone_paths,
            branch_paths=branch_paths,
            cluster_centers=[list(c) for c in backbone.cluster_centers],
        )

    except Exception as e:
        return BackboneResponse(success=False, error=str(e))


def _solve_astar(grid, start, goal, req):
    """A* 求解。"""
    searcher = AStarSDF(
        grid,
        w_sdf=req.w_sdf,
        connectivity=req.connectivity,
    )
    path = searcher.search(start, goal)

    if path is None:
        return RoutingResponse(success=False, error="未找到路径")

    return RoutingResponse(
        success=True,
        result=PathResult(
            path=[list(p) for p in path],
            length=path_length(path),
            turns=count_turns(path),
            method="astar",
        ),
    )


def _solve_ip(grid, start, goal, req):
    """IP 求解。"""
    from ahl.optimization.ortools.model_builder import RoutingModelBuilder

    builder = RoutingModelBuilder(
        grid,
        connectivity=req.connectivity,
        sdf_weight=req.w_sdf,
    )

    if req.max_turns is not None:
        sol = builder.solve_with_turn_limit(start, goal, req.max_turns)
    else:
        sol = builder.solve_shortest_path(start, goal)

    from ahl.optimization.ip.flow_network import SolveStatus
    if sol.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE):
        return RoutingResponse(
            success=True,
            result=PathResult(
                path=[list(p) for p in sol.path],
                length=sol.objective_value,
                turns=sol.turn_count,
                method="ip_scip",
            ),
        )
    else:
        return RoutingResponse(success=False, error="IP 求解无可行解")
