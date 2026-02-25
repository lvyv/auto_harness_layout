import matplotlib.pyplot as plt

import numpy as np
from ortools.sat.python import cp_model


class HarnessRouterGrid:
    def __init__(self, width, height, obstacles, wl=0.1, wb=0.9):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles)
        self.wl = wl
        self.wb = wb
        self.cable_paths = {}

    def _get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) not in self.obstacles:
                    neighbors.append((nx, ny))
        return neighbors

    def save_to_npz(self, filename):
        # 1️⃣ 保存地图 grid
        grid = np.ones((self.height, self.width), dtype=np.uint8)
        for (x, y) in self.obstacles:
            grid[y, x] = 0

        # 2️⃣ 障碍转为数组
        obstacles_array = np.array(list(self.obstacles), dtype=np.int32)

        # 3️⃣ 路径转为可存储结构
        # 结构: dict -> {cid: Nx4数组 (x1,y1,x2,y2)}
        path_dict = {}
        for cid, path in self.cable_paths.items():
            if len(path) > 0:
                path_array = np.array(
                    [[u[0], u[1], v[0], v[1]] for (u, v) in path],
                    dtype=np.int32
                )
            else:
                path_array = np.zeros((0, 4), dtype=np.int32)

            path_dict[cid] = path_array

        # 4️⃣ 保存
        np.savez_compressed(
            filename,
            width=self.width,
            height=self.height,
            grid=grid,
            obstacles=obstacles_array,
            paths=path_dict  # 使用pickle
        )

        print(f"Saved to {filename}")

    def solve_single_cable(self, cable_id, start, end):
        model = cp_model.CpModel()
        edge_vars = {}
        nodes = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.obstacles]

        for u in nodes:
            for v in self._get_neighbors(u):
                edge_vars[(u, v)] = model.NewBoolVar(f'e_{cable_id}_{u}_{v}')

        other_used_edges = set()
        for cid, path in self.cable_paths.items():
            if cid != cable_id: other_used_edges.update(path)

        obj_terms = []
        for (u, v), var in edge_vars.items():
            is_shared = (u, v) in other_used_edges or (v, u) in other_used_edges
            weight = self.wl if is_shared else (self.wl + self.wb)
            obj_terms.append(var * int(weight * 100))

        model.Minimize(sum(obj_terms))

        for node in nodes:
            out_vars = [edge_vars[(node, v)] for v in self._get_neighbors(node)]
            in_vars = [edge_vars[(v, node)] for v in self._get_neighbors(node)]
            if node == start:
                model.Add(sum(out_vars) - sum(in_vars) == 1)
            elif node == end:
                model.Add(sum(out_vars) - sum(in_vars) == -1)
            else:
                model.Add(sum(out_vars) - sum(in_vars) == 0)

        solver = cp_model.CpSolver()
        if solver.Solve(model) in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return [edge for edge, var in edge_vars.items() if solver.Value(var) == 1]
        return []

    def run(self, cables):
        for cid, s, e in cables:
            self.cable_paths[cid] = self.solve_single_cable(cid, s, e)
        for _ in range(3):
            for cid, s, e in cables:
                self.cable_paths[cid] = self.solve_single_cable(cid, s, e)

    def visualize(self):
        # 1. 构建背景网格 (1:白色通路, 0:黑色障碍)
        grid = np.ones((self.height, self.width))
        for (x, y) in self.obstacles:
            grid[y, x] = 0.0

        fig, ax = plt.subplots(figsize=(10, 8))

        # 2. 绘制网格背景
        # 使用 extent 将格子对齐到整数中心
        ax.imshow(grid, cmap='gray', origin='lower', extent=[-0.5, self.width - 0.5, -0.5, self.height - 0.5])

        # 3. 关键修正：设置网格线为格子边界
        # 主刻度显示坐标数字
        ax.set_xticks(np.arange(self.width))
        ax.set_yticks(np.arange(self.height))
        # 次刻度设置在格子边缘 (-0.5, 0.5, 1.5 ...)
        ax.set_xticks(np.arange(self.width + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.height + 1) - 0.5, minor=True)

        # 仅在次刻度位置显示网格线，这样网格线就会完美包围黑白格子
        ax.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', size=0)  # 隐藏次刻度的凸起小线

        # 4. 绘制路径
        # colors = ['#FF0000', '#0000FF']  # 纯红和纯蓝
        colors = ['#FF0000', '#0000FF', '#00AA00']  # 红 蓝 绿
        for i, (cid, path) in enumerate(self.cable_paths.items()):
            offset = (i - 0.5) * 0.12
            for u, v in path:
                ax.plot([u[0] + offset, v[0] + offset],
                        [u[1] + offset, v[1] + offset],
                        color=colors[i % len(colors)],
                        lw=3,
                        solid_capstyle='round',
                        label=cid if (u, v) == path[0] else "")

        # 标记起点和终点
        for cid, path in self.cable_paths.items():
            if path:
                start_node = path[0][0]
                end_node = path[-1][1]
                ax.plot(start_node[0], start_node[1], 'go', markersize=8)  # 起点绿圆
                ax.plot(end_node[0], end_node[1], 'rx', markersize=10, mew=3)  # 终点红叉

        plt.title("Cable Harness Routing (Corrected Grid Alignment)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()


# --- 运行测试 ---
if __name__ == "__main__":

    width = 40
    height = 40

    obstacles = set()

    # ----------------------------
    # 1️⃣ 外围墙体（边界）
    # ----------------------------
    for x in range(width):
        obstacles.add((x, 0))
        obstacles.add((x, height - 1))
    for y in range(height):
        obstacles.add((0, y))
        obstacles.add((width - 1, y))

    # ----------------------------
    # 2️⃣ 中央大矩形障碍（形成绕行）
    # ----------------------------
    for x in range(10, 30):
        for y in range(15, 25):
            obstacles.add((x, y))

    # 留一个狭窄通道
    for y in range(19, 21):
        obstacles.discard((20, y))

    # ----------------------------
    # 3️⃣ 左侧竖向墙（多瓶颈）
    # ----------------------------
    for y in range(5, 35):
        obstacles.add((8, y))

    # 打开几个小孔
    for gap in [10, 20, 30]:
        obstacles.discard((8, gap))

    # ----------------------------
    # 4️⃣ 右侧迷宫结构
    # ----------------------------
    for x in range(25, 38):
        if x % 2 == 0:
            for y in range(5, 35):
                obstacles.add((x, y))

    # 打开几个迷宫通道
    for y in [8, 18, 28]:
        for x in range(25, 38):
            obstacles.discard((x, y))

    # ----------------------------
    # 创建路由器
    # ----------------------------
    router = HarnessRouterGrid(
        width,
        height,
        obstacles,
        wl=0.1,   # 共享奖励
        wb=1.2    # 不共享惩罚
    )

    cables = [
        ('Cable_Red', (2, 2), (37, 37)),
        ('Cable_Blue', (2, 37), (37, 37)),
        ('Cable_Green', (37, 2), (2, 37))
    ]

    router.run(cables)
    router.visualize()