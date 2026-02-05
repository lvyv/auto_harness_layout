import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import networkx as nx
from scipy.spatial import cKDTree

# 方法1：手动设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# ============================
# 1. 读取图像
# ============================
img = cv2.imread("satellite.jpg")  # 替换为你的卫星图像路径
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV读取为BGR，转换为RGB方便显示

plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)
plt.title("原图")
plt.axis('off')
plt.show()

# ============================
# 2. 灰度化
# ============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 8))
plt.imshow(gray, cmap='gray')
plt.title("灰度图")
plt.axis('off')
plt.show()

# ============================
# 3. 去噪
# ============================
blur = cv2.GaussianBlur(gray, (5, 5), 0)

plt.figure(figsize=(8, 8))
plt.imshow(blur, cmap='gray')
plt.title("高斯模糊后")
plt.axis('off')
plt.show()

# ============================
# 4. 二值化
# ============================
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 如果道路是暗的，需要反转
binary = cv2.bitwise_not(binary)

plt.figure(figsize=(8, 8))
plt.imshow(binary, cmap='gray')
plt.title("二值化 + 反转")
plt.axis('off')
plt.show()

# 转布尔类型
binary_bool = binary.astype(bool)

# ============================
# 5. Skeleton 提取
# ============================
skeleton = skeletonize(binary_bool)

plt.figure(figsize=(8, 8))
plt.imshow(skeleton, cmap='gray')
plt.title("骨架化结果")
plt.axis('off')
plt.show()

# ============================
# 5. 构建节点+边网络
# ============================
# 提取骨架坐标
y_idxs, x_idxs = np.nonzero(skeleton)
coords = np.column_stack((x_idxs, y_idxs))

# 用 KDTree 快速找邻居
tree = cKDTree(coords)

# 构建图
G = nx.Graph()
for idx, (x, y) in enumerate(coords):
    G.add_node(idx, pos=(x, y))

    # 找附近像素（8邻域以内）
    neighbors = tree.query_ball_point([x, y], r=1.5)
    for n in neighbors:
        if n != idx:
            G.add_edge(idx, n, weight=np.linalg.norm(coords[idx] - coords[n]))

# ============================
# 6. 可视化骨架 + 网络
# ============================
plt.figure(figsize=(12, 12))
plt.imshow(gray, cmap='gray')
plt.scatter(coords[:, 0], coords[:, 1], s=1, c='red')  # 骨架点
# 绘制边
for u, v in G.edges():
    x0, y0 = G.nodes[u]['pos']
    x1, y1 = G.nodes[v]['pos']
    plt.plot([x0, x1], [y0, y1], 'y-', linewidth=0.5)
plt.title("Skeleton + Road Network Graph")
plt.axis('off')
plt.show()

# ============================
# 7. 示例：路径规划
# ============================
# 随机选择两个节点作为起点和终点
start_node = 0
end_node = len(coords) - 1
path = nx.shortest_path(G, start_node, end_node, weight='weight')

# 绘制路径
plt.figure(figsize=(12, 12))
plt.imshow(gray, cmap='gray')
path_coords = coords[path]
plt.plot(path_coords[:, 0], path_coords[:, 1], 'lime', linewidth=2)
plt.title("Shortest Path on Skeleton Network")
plt.axis('off')
plt.show()
