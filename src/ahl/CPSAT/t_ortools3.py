import sys
import numpy as np
from ortools.sat.python import cp_model
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox,
                             QGroupBox, QGridLayout, QMessageBox, QCheckBox)
from PyQt6.QtCore import Qt, QRectF, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QMouseEvent


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
            paths=path_dict
        )

        print(f"Saved to {filename}")

    def solve_single_cable(self, cable_id, start, end):
        """求解单条电缆路径"""
        # 检查起点和终点是否有效
        if start in self.obstacles or end in self.obstacles:
            print(f"Warning: Start or end point for {cable_id} is blocked by obstacle")
            return []

        model = cp_model.CpModel()
        edge_vars = {}
        nodes = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.obstacles]

        # 如果没有可用节点，返回空路径
        if not nodes:
            return []

        for u in nodes:
            for v in self._get_neighbors(u):
                edge_vars[(u, v)] = model.NewBoolVar(f'e_{cable_id}_{u}_{v}')

        other_used_edges = set()
        for cid, path in self.cable_paths.items():
            if cid != cable_id:
                other_used_edges.update(path)

        obj_terms = []
        for (u, v), var in edge_vars.items():
            is_shared = (u, v) in other_used_edges or (v, u) in other_used_edges
            weight = self.wl if is_shared else (self.wl + self.wb)
            obj_terms.append(var * int(weight * 100))

        if obj_terms:
            model.Minimize(sum(obj_terms))

        # 添加流量守恒约束
        for node in nodes:
            out_vars = [edge_vars[(node, v)] for v in self._get_neighbors(node) if (node, v) in edge_vars]
            in_vars = [edge_vars[(v, node)] for v in self._get_neighbors(node) if (v, node) in edge_vars]

            if node == start:
                model.Add(sum(out_vars) - sum(in_vars) == 1)
            elif node == end:
                model.Add(sum(out_vars) - sum(in_vars) == -1)
            else:
                model.Add(sum(out_vars) - sum(in_vars) == 0)

        solver = cp_model.CpSolver()
        # 设置求解时间限制
        solver.parameters.max_time_in_seconds = 5.0

        status = solver.Solve(model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return [edge for edge, var in edge_vars.items() if solver.Value(var) == 1]
        else:
            print(f"Warning: No solution found for {cable_id}")
            return []

    def run(self, cables):
        """运行路径规划"""
        # 先清除所有路径
        self.cable_paths = {}

        # 第一轮：独立求解每条路径
        success = True
        for cid, s, e in cables:
            path = self.solve_single_cable(cid, s, e)
            if path:
                self.cable_paths[cid] = path
            else:
                success = False
                print(f"Failed to find path for {cid}")

        # 如果第一轮失败，返回
        if not success:
            return False

        # 后续轮次：考虑共享优化
        for iteration in range(3):
            changed = False
            for cid, s, e in cables:
                new_path = self.solve_single_cable(cid, s, e)
                if new_path and new_path != self.cable_paths.get(cid, []):
                    self.cable_paths[cid] = new_path
                    changed = True
            if not changed:
                break

        return True


class ClickableRoutingVisualizer(QWidget):
    """支持点击编辑的可视化组件"""

    # 定义颜色
    COLORS = {
        'Cable_Red': QColor(255, 0, 0),  # 红色
        'Cable_Blue': QColor(0, 0, 255),  # 蓝色
        'Cable_Green': QColor(0, 170, 0),  # 绿色
        'obstacle': QColor(50, 50, 50),  # 深灰色障碍
        'path': QColor(100, 100, 100),  # 灰色路径
        'grid': QColor(200, 200, 200),  # 浅灰色网格
        'background': QColor(255, 255, 255),  # 白色背景
        'start': QColor(0, 255, 0),  # 绿色起点
        'end': QColor(255, 0, 0),  # 红色终点
        'highlight': QColor(255, 255, 0, 100)  # 半透明黄色高亮
    }

    # 自定义信号
    obstacle_changed = pyqtSignal()

    def __init__(self, router):
        super().__init__()
        self.router = router
        self.cell_size = 30  # 每个格子的大小（像素）
        self.offset = 15  # 偏移量用于绘制多条线

        # 编辑模式标志
        self.edit_mode = True
        self.highlighted_cell = None  # 当前鼠标悬停的格子

        # 设置鼠标追踪以获取悬停事件
        self.setMouseTracking(True)

        # 设置窗口大小
        self.setMinimumSize(
            router.width * self.cell_size + 20,
            router.height * self.cell_size + 20
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 绘制背景
        self._draw_background(painter)

        # 绘制网格
        self._draw_grid(painter)

        # 绘制障碍物
        self._draw_obstacles(painter)

        # 绘制高亮格子（鼠标悬停）
        self._draw_highlight(painter)

        # 绘制路径
        self._draw_paths(painter)

        # 绘制起点和终点
        self._draw_endpoints(painter)

    def _draw_background(self, painter):
        """绘制白色背景"""
        painter.fillRect(self.rect(), self.COLORS['background'])

    def _draw_grid(self, painter):
        """绘制网格线"""
        pen = QPen(self.COLORS['grid'], 1, Qt.PenStyle.SolidLine)
        painter.setPen(pen)

        # 绘制垂直线
        for i in range(self.router.width + 1):
            x = i * self.cell_size
            painter.drawLine(x, 0, x, self.router.height * self.cell_size)

        # 绘制水平线
        for i in range(self.router.height + 1):
            y = i * self.cell_size
            painter.drawLine(0, y, self.router.width * self.cell_size, y)

    def _draw_obstacles(self, painter):
        """绘制障碍物（填充黑色格子）"""
        painter.setBrush(QBrush(self.COLORS['obstacle']))
        painter.setPen(Qt.PenStyle.NoPen)

        for x, y in self.router.obstacles:
            rect = QRectF(
                x * self.cell_size + 1,
                y * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            painter.drawRect(rect)

    def _draw_highlight(self, painter):
        """绘制鼠标悬停的高亮格子"""
        if self.highlighted_cell and self.edit_mode:
            x, y = self.highlighted_cell
            painter.setBrush(QBrush(self.COLORS['highlight']))
            painter.setPen(Qt.PenStyle.NoPen)

            rect = QRectF(
                x * self.cell_size + 1,
                y * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            painter.drawRect(rect)

    def _draw_paths(self, painter):
        """绘制电缆路径"""
        # 预定义颜色列表
        colors = [self.COLORS['Cable_Red'],
                  self.COLORS['Cable_Blue'],
                  self.COLORS['Cable_Green']]

        # 计算每条路径的偏移量，使多条路径可以区分
        num_paths = len(self.router.cable_paths)
        if num_paths > 1:
            offsets = np.linspace(-self.offset / 2, self.offset / 2, num_paths)
        else:
            offsets = [0]

        for idx, (cid, path) in enumerate(self.router.cable_paths.items()):
            if not path:
                continue

            # 设置画笔
            pen = QPen(colors[idx % len(colors)], 3, Qt.PenStyle.SolidLine)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)

            offset = offsets[idx]

            # 绘制路径线段
            for u, v in path:
                x1 = u[0] * self.cell_size + self.cell_size / 2 + offset
                y1 = u[1] * self.cell_size + self.cell_size / 2 + offset
                x2 = v[0] * self.cell_size + self.cell_size / 2 + offset
                y2 = v[1] * self.cell_size + self.cell_size / 2 + offset

                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

    def _draw_endpoints(self, painter):
        """绘制起点和终点"""
        for cid, path in self.router.cable_paths.items():
            if not path:
                continue

            # 绘制起点（绿色圆点）
            start_node = path[0][0]
            x = start_node[0] * self.cell_size + self.cell_size / 2
            y = start_node[1] * self.cell_size + self.cell_size / 2

            painter.setBrush(QBrush(self.COLORS['start']))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(x - 6), int(y - 6), 12, 12)

            # 绘制终点（红色X）
            end_node = path[-1][1]
            x = end_node[0] * self.cell_size + self.cell_size / 2
            y = end_node[1] * self.cell_size + self.cell_size / 2

            pen = QPen(self.COLORS['end'], 3, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # 绘制X
            painter.drawLine(int(x - 6), int(y - 6), int(x + 6), int(y + 6))
            painter.drawLine(int(x - 6), int(y + 6), int(x + 6), int(y - 6))

    def _get_cell_from_pos(self, pos):
        """从像素坐标获取格子坐标"""
        x = int(pos.x() / self.cell_size)
        y = int(pos.y() / self.cell_size)

        # 检查是否在有效范围内
        if 0 <= x < self.router.width and 0 <= y < self.router.height:
            return (x, y)
        return None

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件 - 更新高亮格子"""
        if self.edit_mode:
            cell = self._get_cell_from_pos(event.pos())
            if cell != self.highlighted_cell:
                self.highlighted_cell = cell
                self.update()  # 触发重绘

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标点击事件 - 添加/删除障碍"""
        if not self.edit_mode:
            return

        cell = self._get_cell_from_pos(event.pos())
        if cell is None:
            return

        x, y = cell

        # 检查是否是起点或终点（不能设置为障碍）
        is_endpoint = False
        for cid, path in self.router.cable_paths.items():
            if path:
                if path[0][0] == (x, y) or path[-1][1] == (x, y):
                    is_endpoint = True
                    break

        if is_endpoint:
            QMessageBox.information(self, "Cannot Edit",
                                    "Cannot add/remove obstacles at start or end points!")
            return

        # 左键添加障碍
        if event.button() == Qt.MouseButton.LeftButton:
            if (x, y) not in self.router.obstacles:
                # 检查是否是边界（可选：允许编辑边界）
                # 这里允许编辑所有格子，包括边界
                self.router.obstacles.add((x, y))
                self.obstacle_changed.emit()
                self.update()

        # 右键消除障碍
        elif event.button() == Qt.MouseButton.RightButton:
            if (x, y) in self.router.obstacles:
                self.router.obstacles.remove((x, y))
                self.obstacle_changed.emit()
                self.update()

    def leaveEvent(self, event):
        """鼠标离开窗口时清除高亮"""
        self.highlighted_cell = None
        self.update()

    def set_edit_mode(self, enabled):
        """设置编辑模式"""
        self.edit_mode = enabled
        if not enabled:
            self.highlighted_cell = None
        self.update()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self, router, cables):
        super().__init__()
        self.router = router
        self.cables = cables
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Cable Harness Routing - Interactive Editor")
        self.setGeometry(100, 100, 900, 800)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧：可视化区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.visualizer = ClickableRoutingVisualizer(self.router)
        self.visualizer.obstacle_changed.connect(self.on_obstacle_changed)
        left_layout.addWidget(self.visualizer)

        main_layout.addWidget(left_widget, 2)

        # 右侧：控制面板
        right_widget = QWidget()
        right_widget.setMaximumWidth(300)
        right_layout = QVBoxLayout(right_widget)

        # 信息组
        info_group = QGroupBox("Grid Information")
        info_layout = QGridLayout()

        info_layout.addWidget(QLabel("Width:"), 0, 0)
        info_layout.addWidget(QLabel(str(self.router.width)), 0, 1)
        info_layout.addWidget(QLabel("Height:"), 1, 0)
        info_layout.addWidget(QLabel(str(self.router.height)), 1, 1)
        info_layout.addWidget(QLabel("Obstacles:"), 2, 0)
        self.obstacle_count_label = QLabel(str(len(self.router.obstacles)))
        info_layout.addWidget(self.obstacle_count_label, 2, 1)

        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)

        # 编辑控制组
        edit_group = QGroupBox("Edit Controls")
        edit_layout = QVBoxLayout()

        self.edit_mode_checkbox = QCheckBox("Enable Edit Mode")
        self.edit_mode_checkbox.setChecked(True)
        self.edit_mode_checkbox.stateChanged.connect(self.toggle_edit_mode)
        edit_layout.addWidget(self.edit_mode_checkbox)

        edit_layout.addWidget(QLabel("Left Click: Add Obstacle"))
        edit_layout.addWidget(QLabel("Right Click: Remove Obstacle"))
        edit_layout.addWidget(QLabel("Note: Cannot edit start/end points"))

        edit_group.setLayout(edit_layout)
        right_layout.addWidget(edit_group)

        # 路径控制组
        path_group = QGroupBox("Path Controls")
        path_layout = QVBoxLayout()

        # 电缆信息
        for cid, start, end in self.cables:
            cable_label = QLabel(f"{cid}: ({start[0]},{start[1]}) → ({end[0]},{end[1]})")
            path_layout.addWidget(cable_label)

        # 按钮
        recalc_btn = QPushButton("Recalculate Paths")
        recalc_btn.clicked.connect(self.recalculate_paths)
        recalc_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px; }")
        path_layout.addWidget(recalc_btn)

        clear_btn = QPushButton("Clear All Obstacles")
        clear_btn.clicked.connect(self.clear_obstacles)
        clear_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 8px; }")
        path_layout.addWidget(clear_btn)

        path_group.setLayout(path_layout)
        right_layout.addWidget(path_group)

        # 状态组
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)

        # 退出按钮
        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(self.close)
        exit_btn.setStyleSheet("QPushButton { background-color: #666; color: white; padding: 8px; }")
        right_layout.addWidget(exit_btn)

        right_layout.addStretch()
        main_layout.addWidget(right_widget, 1)

        # 初始计算路径
        self.recalculate_paths()

    def toggle_edit_mode(self, state):
        """切换编辑模式"""
        enabled = state == Qt.CheckState.Checked.value
        self.visualizer.set_edit_mode(enabled)

    def on_obstacle_changed(self):
        """障碍物改变时的处理"""
        self.obstacle_count_label.setText(str(len(self.router.obstacles)))
        self.status_label.setText("Obstacles changed. Click 'Recalculate Paths' to update routes.")

    def recalculate_paths(self):
        """重新计算路径"""
        self.status_label.setText("Calculating paths...")
        self.status_label.repaint()  # 强制更新显示
        QApplication.processEvents()  # 处理事件以更新UI

        # 运行路径规划
        success = self.router.run(self.cables)

        if success:
            self.status_label.setText("Paths calculated successfully!")
        else:
            self.status_label.setText("Warning: Some paths could not be found!")

        # 刷新显示
        self.visualizer.update()

    def clear_obstacles(self):
        """清除所有障碍物（除了边界）"""
        # 保留边界作为障碍物
        boundary_obstacles = set()
        for x in range(self.router.width):
            boundary_obstacles.add((x, 0))
            boundary_obstacles.add((x, self.router.height - 1))
        for y in range(self.router.height):
            boundary_obstacles.add((0, y))
            boundary_obstacles.add((self.router.width - 1, y))

        self.router.obstacles = boundary_obstacles
        self.obstacle_count_label.setText(str(len(self.router.obstacles)))
        self.visualizer.update()
        self.status_label.setText("Obstacles cleared. Click 'Recalculate Paths' to update.")


# --- 运行测试 ---
if __name__ == "__main__":
    width = 15  # 减小网格大小以便于编辑
    height = 15

    # 初始障碍物（边界）
    obstacles = set()
    for x in range(width):
        obstacles.add((x, 0))
        obstacles.add((x, height - 1))
    for y in range(height):
        obstacles.add((0, y))
        obstacles.add((width - 1, y))

    # 添加一些内部障碍物作为示例
    obstacles.add((5, 5))
    obstacles.add((5, 6))
    obstacles.add((6, 5))
    obstacles.add((8, 8))
    obstacles.add((8, 9))
    obstacles.add((9, 8))

    # 创建路由器
    router = HarnessRouterGrid(
        width,
        height,
        obstacles,
        wl=0.1,  # 共享奖励
        wb=1.2  # 不共享惩罚
    )

    # 定义电缆起点终点
    cables = [
        ('Cable_Red', (2, 2), (7, 7)),
        ('Cable_Blue', (3, 3), (10, 10)),
        ('Cable_Green', (4, 4), (9, 11))
    ]

    # 创建PyQt6应用
    app = QApplication(sys.argv)
    window = MainWindow(router, cables)
    window.show()
    sys.exit(app.exec())