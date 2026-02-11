from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt

# =========================
# 1. 网格定义
# =========================
W, H = 16, 16
start = (0, 0)
goal  = (4, 7)

obstacles = {
    (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
    (3, 2),
    (5, 1), (5, 9),
    (6, 1),(6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
}

DIRS = [(1,0), (-1,0), (0,1), (0,-1)]

def inside(xx, yy):
    return 0 <= xx < W and 0 <= yy < H

# =========================
# 2. 构图
# =========================
edges = []
for x in range(W):
    for y in range(H):
        if (x, y) in obstacles:
            continue
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if inside(nx, ny) and (nx, ny) not in obstacles:
                edges.append(((x, y), (nx, ny)))

# =========================
# 3. OR-Tools 求解器
# =========================
solver = pywraplp.Solver.CreateSolver("SCIP")

x = {}
for u, v in edges:
    x[(u, v)] = solver.BoolVar(f"x_{u}_{v}")

# =========================
# 4. 流守恒约束
# =========================
for i in range(W):
    for j in range(H):
        if (i, j) in obstacles:
            continue

        inflow = []
        outflow = []

        for (u, v), var in x.items():
            if v == (i, j):
                inflow.append(var)
            if u == (i, j):
                outflow.append(var)

        if (i, j) == start:
            solver.Add(sum(outflow) - sum(inflow) == 1)
        elif (i, j) == goal:
            solver.Add(sum(outflow) - sum(inflow) == -1)
        else:
            solver.Add(sum(outflow) - sum(inflow) == 0)

# =========================
# 5. 目标函数
# =========================
solver.Minimize(solver.Sum(x[e] for e in x))

# =========================
# 6. 求解
# =========================
status = solver.Solve()

# =========================
# 7. 提取路径
# =========================
selected_edges = {}
for (u, v), var in x.items():
    if var.solution_value() > 0.5:
        selected_edges[u] = v

# 串成路径
path = [start]
cur = start
while cur != goal:
    cur = selected_edges[cur]
    path.append(cur)

print("Path:", path)

# =========================
# 8. 绘图
# =========================
fig, ax = plt.subplots(figsize=(6, 6))
# 设置窗口标题
if fig.canvas.manager is not None:
    fig.canvas.manager.set_window_title("SCIP-求带约束的路径问题")

# 画网格
for x in range(W + 1):
    ax.plot([x, x], [0, H], linewidth=0.5)
for y in range(H + 1):
    ax.plot([0, W], [y, y], linewidth=0.5)

# 障碍
for (x, y) in obstacles:
    ax.add_patch(
        plt.Rectangle((x, y), 1, 1)
    )

# 路径
px = [p[0] + 0.5 for p in path]
py = [p[1] + 0.5 for p in path]
ax.plot(px, py, linewidth=3)

# 起点 & 终点
ax.scatter(start[0] + 0.5, start[1] + 0.5, s=100)
ax.scatter(goal[0] + 0.5, goal[1] + 0.5, s=100)

ax.set_aspect("equal")
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.invert_yaxis()   # 更像矩阵/图像坐标
ax.set_title("OR-Tools IP Shortest Path")

plt.show()
