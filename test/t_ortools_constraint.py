from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt

# =========================
# 1. 网格定义
# =========================
W, H = 16, 16
start = (0, 0)
goal = (4, 7)
MAX_TURNS = 5  # <--- 新增：限制最大转弯次数

obstacles = {
    (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
    (3, 2), (5, 1), (5, 9),
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
}

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def inside(xx, yy):
    return 0 <= xx < W and 0 <= yy < H


# =========================
# 2. 构图与变量初始化
# =========================
solver = pywraplp.Solver.CreateSolver("SCIP")
x = {}
nodes = []
for r in range(W):
    for c in range(H):
        if (r, c) not in obstacles:
            nodes.append((r, c))

for u in nodes:
    for dx, dy in DIRS:
        v = (u[0] + dx, u[1] + dy)
        if v in nodes:
            x[(u, v)] = solver.BoolVar(f"x_{u}_{v}")

# =========================
# 3. 流守恒约束 (基础路径)
# =========================
for node in nodes:
    inflow = [x[e] for e in x if e[1] == node]
    outflow = [x[e] for e in x if e[0] == node]

    if node == start:
        solver.Add(sum(outflow) - sum(inflow) == 1)
    elif node == goal:
        solver.Add(sum(outflow) - sum(inflow) == -1)
    else:
        solver.Add(sum(outflow) - sum(inflow) == 0)
        # 限制每个点最多进出一次，防止环路（虽然最短路通常不产生环，但带约束时建议加上）
        # solver.Add(sum(outflow) <= 1)

# =========================
# 4. 转弯约束 (关键新增)
# =========================
turn_vars = []
for v in nodes:
    if v == start or v == goal:
        continue

    # 找到所有进入 v 和离开 v 的边对
    in_edges = [e for e in x if e[1] == v]
    out_edges = [e for e in x if e[0] == v]

    for e_in in in_edges:
        for e_out in out_edges:
            u = e_in[0]
            w = e_out[1]

            # 判断是否拐弯: 如果 u, v, w 不在一条直线上
            # 向量 (v-u) 和 (w-v)
            vec1 = (v[0] - u[0], v[1] - u[1])
            vec2 = (w[0] - v[0], w[1] - v[1])

            if vec1 != vec2:
                # 定义转弯变量：如果 e_in 和 e_out 同时被选中，则 turn = 1
                t = solver.BoolVar(f"turn_{u}_{v}_{w}")
                # 线性化: t >= x_in + x_out - 1
                solver.Add(t >= x[e_in] + x[e_out] - 1)
                turn_vars.append(t)

# 限制总转弯次数
solver.Add(sum(turn_vars) == MAX_TURNS)

# =========================
# 5. 目标函数 (最短距离)
# =========================
solver.Minimize(solver.Sum(x.values()))

# =========================
# 6. 求解与绘图 (同原代码)
# =========================
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    selected_edges = {u: v for (u, v), var in x.items() if var.solution_value() > 0.5}
    path = [start]
    while path[-1] != goal:
        path.append(selected_edges[path[-1]])

    print(f"Path length: {len(path) - 1}, Turns: {sum(t.solution_value() for t in turn_vars)}")

    # 绘图部分保持不变...
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(W + 1): ax.plot([i, i], [0, H], color='gray', lw=0.5)
    for i in range(H + 1): ax.plot([0, W], [i, i], color='gray', lw=0.5)
    for (ox, oy) in obstacles: ax.add_patch(plt.Rectangle((ox, oy), 1, 1, color='black'))
    px, py = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in path])
    ax.plot(px, py, linewidth=3, color='red')
    ax.scatter(start[0] + 0.5, start[1] + 0.5, color='green', s=100)
    ax.scatter(goal[0] + 0.5, goal[1] + 0.5, color='blue', s=100)
    ax.invert_yaxis()
    plt.show()
else:
    print("No solution found.")