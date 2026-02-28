import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label


def extract_connected_components(trunk_mask, connectivity=4):
    """
    trunk_mask: 0/1 numpy array
    connectivity: 4 或 8
    """

    if connectivity == 4:
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]])
    elif connectivity == 8:
        structure = np.ones((3,3))
    else:
        raise ValueError("connectivity must be 4 or 8")

    labeled, num_components = label(trunk_mask, structure=structure)

    return labeled, num_components

def split_components(labeled, num_components):
    components = []
    for i in range(1, num_components + 1):
        coords = np.argwhere(labeled == i)
        components.append(coords)
    return components

# =====================================================
# 1. 构造测试 grid
# =====================================================

def create_grid(n=200, obstacle_ratio=0.2, seed=0):
    np.random.seed(seed)
    grid = np.zeros((n, n), dtype=np.int8)
    obstacle_mask = np.random.rand(n, n) < obstacle_ratio
    grid[obstacle_mask] = 1
    return grid


# =====================================================
# 2. A* 带扰动
# =====================================================

def astar_with_noise(grid, start, goal, epsilon=0.01):
    n, m = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = current[0] + dx, current[1] + dy

            if 0 <= nx < n and 0 <= ny < m and grid[nx, ny] == 0:
                neighbor = (nx, ny)
                noise = epsilon * np.random.rand()
                tentative_g = g_score[current] + 1 + noise

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    return None


# =====================================================
# 3. 随机采样 OD
# =====================================================

def sample_free_point(grid):
    n = grid.shape[0]
    while True:
        x, y = np.random.randint(0, n), np.random.randint(0, n)
        if grid[x, y] == 0:
            return (x, y)


# =====================================================
# 4. 主流程
# =====================================================

def monte_carlo_trunk_extraction(
    n=200,
    obstacle_ratio=0.2,
    num_samples=2000,
    epsilon=0.05,
    smooth_sigma=2,
    trunk_percentile=90
):
    grid = create_grid(n, obstacle_ratio)
    weight = np.zeros_like(grid, dtype=np.float32)

    for i in range(num_samples):
        start = sample_free_point(grid)
        goal = sample_free_point(grid)

        path = astar_with_noise(grid, start, goal, epsilon)

        if path is not None:
            for (x, y) in path:
                weight[x, y] += 1

        if (i+1) % 200 == 0:
            print(f"Sample {i+1}/{num_samples}")

    # 平滑
    weight_smooth = gaussian_filter(weight, sigma=smooth_sigma)

    # 提取高权重区域
    threshold = np.percentile(weight_smooth[grid == 0], trunk_percentile)
    trunk = (weight_smooth >= threshold).astype(np.int8)

    return grid, weight, weight_smooth, trunk


# =====================================================
# 5. 运行实验
# =====================================================

if __name__ == "__main__":
    grid, weight, weight_smooth, trunk = monte_carlo_trunk_extraction(
        n=200,
        obstacle_ratio=0.25,
        num_samples=3000,
        epsilon=0.05,
        smooth_sigma=2,
        trunk_percentile=92
    )

    # 连通子图
    labeled, num = extract_connected_components(trunk)
    comp = split_components(labeled, num)

    plt.imshow(labeled, cmap='nipy_spectral')
    plt.colorbar()
    plt.title(f"{num} components")
    plt.show()

    # 对比
    plt.figure(figsize=(18,5))
    plt.subplot(1,4,1)
    plt.title("Grid")
    plt.imshow(grid, cmap="gray")

    plt.subplot(1,4,2)
    plt.title("Raw Weight")
    plt.imshow(weight, cmap="hot")

    plt.subplot(1,4,3)
    plt.title("Smoothed Weight")
    plt.imshow(weight_smooth, cmap="hot")

    plt.subplot(1,4,4)
    plt.title("Extracted Trunk")
    plt.imshow(trunk, cmap="gray")

    plt.tight_layout()



    plt.show()