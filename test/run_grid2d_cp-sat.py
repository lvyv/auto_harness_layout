import sys
import json
import os
import numpy as np
from ortools.sat.python import cp_model
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox,
                             QGroupBox, QGridLayout, QMessageBox, QCheckBox,
                             QFileDialog, QListWidget, QLineEdit,
                             QDialog, QDialogButtonBox, QFormLayout)
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QMouseEvent, QAction

from PyQt6.QtCore import QThread, pyqtSignal

# 解决debug的崩溃问题
class SolverThread(QThread):
    finished_signal = pyqtSignal(bool)

    def __init__(self, router, cables):
        super().__init__()
        self.router = router
        self.cables = cables

    def run(self):
        try:
            success = self.router.run(self.cables)
        except Exception as e:
            print("Solver crashed:", e)
            success = False
        self.finished_signal.emit(success)

class CableConfigDialog(QDialog):
    """电缆配置对话框"""

    def __init__(self, parent=None, width=15, height=15, edit_mode=False):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.edit_mode = edit_mode
        self.setWindowTitle("Cable Configuration")
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)

        # 电缆ID
        self.id_edit = QLineEdit()
        if not self.edit_mode:
            self.id_edit.setPlaceholderText("e.g., Cable_Red")
        layout.addRow("Cable ID:", self.id_edit)

        # 起点坐标
        start_layout = QHBoxLayout()
        self.start_x = QSpinBox()
        self.start_x.setRange(0, self.width - 1)
        self.start_y = QSpinBox()
        self.start_y.setRange(0, self.height - 1)
        start_layout.addWidget(QLabel("X:"))
        start_layout.addWidget(self.start_x)
        start_layout.addWidget(QLabel("Y:"))
        start_layout.addWidget(self.start_y)
        layout.addRow("Start Point:", start_layout)

        # 终点坐标
        end_layout = QHBoxLayout()
        self.end_x = QSpinBox()
        self.end_x.setRange(0, self.width - 1)
        self.end_y = QSpinBox()
        self.end_y.setRange(0, self.height - 1)
        end_layout.addWidget(QLabel("X:"))
        end_layout.addWidget(self.end_x)
        end_layout.addWidget(QLabel("Y:"))
        end_layout.addWidget(self.end_y)
        layout.addRow("End Point:", end_layout)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_cable_data(self):
        """获取电缆数据"""
        return {
            'id': self.id_edit.text(),
            'start': (self.start_x.value(), self.start_y.value()),
            'end': (self.end_x.value(), self.end_y.value())
        }

    def set_cable_data(self, cable_data):
        """设置电缆数据"""
        self.id_edit.setText(cable_data['id'])
        self.start_x.setValue(cable_data['start'][0])
        self.start_y.setValue(cable_data['start'][1])
        self.end_x.setValue(cable_data['end'][0])
        self.end_y.setValue(cable_data['end'][1])


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
        """保存到NPZ文件"""
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

    def save_to_json(self, filename):
        """保存到JSON文件（包含所有配置）"""
        # data = {
        #     'width': self.width,
        #     'height': self.height,
        #     'obstacles': list(self.obstacles),
        #     'cable_paths': {},
        #     'wl': self.wl,
        #     'wb': self.wb
        # }
        data = {
            'width': self.width,
            'height': self.height,
            'obstacles': list(self.obstacles),
            'cables_config': cables,  # 保存原始配置：[(id, start, end), ...]
            'cable_paths': {},
            'wl': self.wl,
            'wb': self.wb
        }
        # 保存路径
        for cid, path in self.cable_paths.items():
            path_data = []
            for u, v in path:
                path_data.append({
                    'from': [int(u[0]), int(u[1])],
                    'to': [int(v[0]), int(v[1])]
                })
            data['cable_paths'][cid] = path_data

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, filename):
        """从JSON文件加载"""
        with open(filename, 'r') as f:
            data = json.load(f)

        self.width = data['width']
        self.height = data['height']
        self.obstacles = set(tuple(obs) for obs in data['obstacles'])
        self.wl = data.get('wl', 0.1)
        self.wb = data.get('wb', 0.9)

        # 加载路径
        self.cable_paths = {}
        for cid, path_data in data['cable_paths'].items():
            path = []
            for segment in path_data:
                u = tuple(segment['from'])
                v = tuple(segment['to'])
                path.append((u, v))
            self.cable_paths[cid] = path

        return data.get('cables_config', [])

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
        'Cable_Yellow': QColor(255, 255, 0),  # 黄色
        'Cable_Purple': QColor(128, 0, 128),  # 紫色
        'Cable_Orange': QColor(255, 165, 0),  # 橙色
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
    cable_selected = pyqtSignal(str)

    def __init__(self, router):
        super().__init__()
        self.router = router
        self.cell_size = 16 # 每个格子的大小（像素）
        self.offset = 5  # 偏移量用于绘制多条线

        # 编辑模式标志
        self.edit_mode = True
        self.highlighted_cell = None  # 当前鼠标悬停的格子
        self.selected_cable = None  # 当前选中的电缆

        # 预定义颜色列表（按名称映射）
        self.color_map = {
            'Cable_Red': self.COLORS['Cable_Red'],
            'Cable_Blue': self.COLORS['Cable_Blue'],
            'Cable_Green': self.COLORS['Cable_Green'],
            'Cable_Yellow': self.COLORS['Cable_Yellow'],
            'Cable_Purple': self.COLORS['Cable_Purple'],
            'Cable_Orange': self.COLORS['Cable_Orange']
        }

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
        # 计算每条路径的偏移量，使多条路径可以区分
        num_paths = len(self.router.cable_paths)
        if num_paths > 1:
            offsets = np.linspace(-self.offset / 2, self.offset / 2, num_paths)
        else:
            offsets = [0]

        for idx, (cid, path) in enumerate(self.router.cable_paths.items()):
            if not path:
                continue

            # 获取电缆颜色，如果没有定义则使用默认颜色
            color = self.color_map.get(cid, QColor(100, 100, 100))

            # 如果被选中，加粗显示
            pen_width = 5 if cid == self.selected_cable else 3

            # 设置画笔
            pen = QPen(color, pen_width, Qt.PenStyle.SolidLine)
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

    # def _draw_endpoints(self, painter):
    #     """绘制起点和终点"""
    #     for cid, path in self.router.cable_paths.items():
    #         if not path:
    #             continue
    #
    #         # 获取颜色用于边框
    #         color = self.color_map.get(cid, QColor(100, 100, 100))
    #
    #         # 绘制起点（绿色圆点，带颜色边框）
    #         start_node = path[0][0]
    #         x = start_node[0] * self.cell_size + self.cell_size / 2
    #         y = start_node[1] * self.cell_size + self.cell_size / 2
    #
    #         painter.setBrush(QBrush(self.COLORS['start']))
    #         painter.setPen(QPen(color, 2))
    #         painter.drawEllipse(int(x - 7), int(y - 7), 14, 14)
    #
    #         # 绘制终点（红色X，带颜色边框）
    #         end_node = path[-1][1]
    #         x = end_node[0] * self.cell_size + self.cell_size / 2
    #         y = end_node[1] * self.cell_size + self.cell_size / 2
    #
    #         painter.setPen(QPen(color, 3))
    #         painter.setBrush(Qt.BrushStyle.NoBrush)
    #
    #         # 绘制X
    #         painter.drawLine(int(x - 7), int(y - 7), int(x + 7), int(y + 7))
    #         painter.drawLine(int(x - 7), int(y + 7), int(x + 7), int(y - 7))
    def _draw_endpoints(self, painter):
        """绘制起点和终点（直接从配置数据读取，确保路径失败也能显示）"""
        # 获取主窗口中的原始电缆列表
        # 注意：这里需要确保 visualizer 能访问到 cables 数据，
        # 建议在 MainWindow 初始化 visualizer 后将 self.cables 引用传给它
        # 或者通过 parent 访问。这里假设我们通过 self.router 结构或外部传入获取。

        # 优化建议：在 MainWindow 中：self.visualizer.cables_ref = self.cables
        if not hasattr(self, 'cables_ref') or not self.cables_ref:
            return

        for cid, start_node, end_node in self.cables_ref:
            # 获取颜色
            color = self.color_map.get(cid, QColor(100, 100, 100))

            # --- 绘制起点（绿圆） ---
            sx = start_node[0] * self.cell_size + self.cell_size / 2
            sy = start_node[1] * self.cell_size + self.cell_size / 2

            painter.setBrush(QBrush(self.COLORS['start']))
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(int(sx - 8), int(sy - 8), 16, 16)  # 稍微加大一点

            # 绘制 ID 文字标签，方便辨认
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawText(int(sx + 10), int(sy - 10), cid)

            # --- 绘制终点（红X） ---
            ex = end_node[0] * self.cell_size + self.cell_size / 2
            ey = end_node[1] * self.cell_size + self.cell_size / 2

            painter.setPen(QPen(color, 4))  # 加粗X
            painter.setBrush(Qt.BrushStyle.NoBrush)

            delta = 7
            painter.drawLine(int(ex - delta), int(ey - delta), int(ex + delta), int(ey + delta))
            painter.drawLine(int(ex - delta), int(ey + delta), int(ex + delta), int(ey - delta))

    def _get_cell_from_pos(self, pos):
        """从像素坐标获取格子坐标"""
        x = int(pos.x() / self.cell_size)
        y = int(pos.y() / self.cell_size)

        # 检查是否在有效范围内
        if 0 <= x < self.router.width and 0 <= y < self.router.height:
            return (x, y)
        return None

    def _get_cable_at_pos(self, pos):
        """获取点击位置附近的电缆"""
        x, y = pos.x(), pos.y()
        cell_x, cell_y = int(x / self.cell_size), int(y / self.cell_size)

        # 检查每个电缆路径
        for cid, path in self.router.cable_paths.items():
            for u, v in path:
                # 计算线段的中点
                mid_x = (u[0] + v[0]) / 2 * self.cell_size + self.cell_size / 2
                mid_y = (u[1] + v[1]) / 2 * self.cell_size + self.cell_size / 2

                # 检查点击位置是否接近线段
                distance = ((x - mid_x) ** 2 + (y - mid_y) ** 2) ** 0.5
                if distance < 10:  # 10像素的容差
                    return cid
        return None

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件 - 更新高亮格子"""
        if self.edit_mode:
            cell = self._get_cell_from_pos(event.pos())
            if cell != self.highlighted_cell:
                self.highlighted_cell = cell
                self.update()  # 触发重绘

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标点击事件"""
        # 检查是否点击到电缆（用于选择）
        if event.button() == Qt.MouseButton.LeftButton and not self.edit_mode:
            cable_id = self._get_cable_at_pos(event.pos())
            if cable_id:
                self.selected_cable = cable_id
                self.cable_selected.emit(cable_id)
                self.update()
                return

        # 编辑模式下的障碍物编辑
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

    def set_selected_cable(self, cable_id):
        """设置选中的电缆"""
        self.selected_cable = cable_id
        self.update()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self, router, cables):
        super().__init__()
        self.router = router
        self.cables = cables
        self.current_file = None
        self.init_ui()
        self.create_menu()
        # 初始计算路径
        # self.recalculate_paths()

    def init_ui(self):
        self.setWindowTitle("Cable Harness Routing - Interactive Editor")
        self.setGeometry(100, 100, 1000, 800)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧：可视化区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.visualizer = ClickableRoutingVisualizer(self.router)
        self.visualizer.cables_ref = self.cables  # <--- 添加这一行，建立数据引用
        self.visualizer.obstacle_changed.connect(self.on_obstacle_changed)

        # self.visualizer = ClickableRoutingVisualizer(self.router)
        # self.visualizer.obstacle_changed.connect(self.on_obstacle_changed)
        self.visualizer.cable_selected.connect(self.on_cable_selected)
        left_layout.addWidget(self.visualizer)

        main_layout.addWidget(left_widget, 2)

        # 右侧：控制面板
        right_widget = QWidget()
        right_widget.setMaximumWidth(350)
        right_layout = QVBoxLayout(right_widget)

        # 文件操作组
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()

        # 保存按钮
        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self.save_configuration)
        save_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 8px; }")
        file_layout.addWidget(save_btn)

        # 载入按钮
        load_btn = QPushButton("Load Configuration")
        load_btn.clicked.connect(self.load_configuration)
        load_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px; }")
        file_layout.addWidget(load_btn)

        file_group.setLayout(file_layout)
        right_layout.addWidget(file_group)

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
        info_layout.addWidget(QLabel("Cables:"), 3, 0)
        self.cable_count_label = QLabel(str(len(self.cables)))
        info_layout.addWidget(self.cable_count_label, 3, 1)

        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)

        # 电缆管理组
        cable_group = QGroupBox("Cable Management")
        cable_layout = QVBoxLayout()

        # 电缆列表
        self.cable_list = QListWidget()
        self.cable_list.itemClicked.connect(self.on_cable_list_clicked)
        self.update_cable_list()
        cable_layout.addWidget(QLabel("Cables:"))
        cable_layout.addWidget(self.cable_list)

        # 电缆操作按钮
        cable_btn_layout = QHBoxLayout()

        add_cable_btn = QPushButton("Add Cable")
        add_cable_btn.clicked.connect(self.add_cable)
        cable_btn_layout.addWidget(add_cable_btn)

        edit_cable_btn = QPushButton("Edit Cable")
        edit_cable_btn.clicked.connect(self.edit_cable)
        cable_btn_layout.addWidget(edit_cable_btn)

        remove_cable_btn = QPushButton("Remove")
        remove_cable_btn.clicked.connect(self.remove_cable)
        cable_btn_layout.addWidget(remove_cable_btn)

        cable_layout.addLayout(cable_btn_layout)

        cable_group.setLayout(cable_layout)
        right_layout.addWidget(cable_group)

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

        # 按钮
        recalc_btn = QPushButton("Recalculate Paths")
        recalc_btn.clicked.connect(self.recalculate_paths)
        recalc_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px; }")
        path_layout.addWidget(recalc_btn)

        clear_obstacles_btn = QPushButton("Clear All Obstacles")
        clear_obstacles_btn.clicked.connect(self.clear_obstacles)
        clear_obstacles_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 8px; }")
        path_layout.addWidget(clear_obstacles_btn)

        clear_paths_btn = QPushButton("Clear All Paths")
        clear_paths_btn.clicked.connect(self.clear_paths)
        clear_paths_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; padding: 8px; }")
        path_layout.addWidget(clear_paths_btn)

        path_group.setLayout(path_layout)
        right_layout.addWidget(path_group)

        # 状态组
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        status_layout.addWidget(self.file_label)

        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)

        right_layout.addStretch()
        main_layout.addWidget(right_widget, 1)


    def create_menu(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('File')

        save_action = QAction('Save Configuration', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_configuration)
        file_menu.addAction(save_action)

        load_action = QAction('Load Configuration', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_configuration)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        export_action = QAction('Export as NPZ', self)
        export_action.triggered.connect(self.export_npz)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 编辑菜单
        edit_menu = menubar.addMenu('Edit')

        toggle_edit_action = QAction('Toggle Edit Mode', self)
        toggle_edit_action.setShortcut('Ctrl+E')
        toggle_edit_action.triggered.connect(self.toggle_edit_mode_menu)
        edit_menu.addAction(toggle_edit_action)

        # 视图菜单
        view_menu = menubar.addMenu('View')

        zoom_in_action = QAction('Zoom In', self)
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction('Zoom Out', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)

    def toggle_edit_mode(self, state):
        """切换编辑模式"""
        enabled = state == Qt.CheckState.Checked.value
        self.visualizer.set_edit_mode(enabled)

    def toggle_edit_mode_menu(self):
        """菜单切换编辑模式"""
        current = self.edit_mode_checkbox.isChecked()
        self.edit_mode_checkbox.setChecked(not current)

    def zoom_in(self):
        """放大视图"""
        self.visualizer.cell_size = min(50, self.visualizer.cell_size + 5)
        self.visualizer.update()

    def zoom_out(self):
        """缩小视图"""
        self.visualizer.cell_size = max(15, self.visualizer.cell_size - 5)
        self.visualizer.update()

    def on_obstacle_changed(self):
        """障碍物改变时的处理"""
        self.obstacle_count_label.setText(str(len(self.router.obstacles)))
        self.status_label.setText("Obstacles changed. Click 'Recalculate Paths' to update routes.")

    def on_cable_selected(self, cable_id):
        """电缆被选中时的处理"""
        self.status_label.setText(f"Selected: {cable_id}")

        # 在列表中高亮
        for i in range(self.cable_list.count()):
            item = self.cable_list.item(i)
            if cable_id in item.text():
                self.cable_list.setCurrentItem(item)
                break

    def on_cable_list_clicked(self, item):
        """点击电缆列表时的处理"""
        cable_id = item.text().split(' ')[0]  # 获取电缆ID
        self.visualizer.set_selected_cable(cable_id)

    def update_cable_list(self):
        """更新电缆列表"""
        self.cable_list.clear()
        for cid, start, end in self.cables:
            self.cable_list.addItem(f"{cid} : ({start[0]},{start[1]}) → ({end[0]},{end[1]})")
        self.cable_count_label.setText(str(len(self.cables)))

    def add_cable(self):
        """添加新电缆"""
        dialog = CableConfigDialog(self, self.router.width, self.router.height)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_cable_data()
            if data['id']:
                # 检查ID是否已存在
                if any(cid == data['id'] for cid, _, _ in self.cables):
                    QMessageBox.warning(self, "Duplicate ID", f"Cable ID '{data['id']}' already exists!")
                    return

                self.cables.append((data['id'], data['start'], data['end']))
                self.update_cable_list()
                self.status_label.setText(f"Added cable: {data['id']}")

    def edit_cable(self):
        """编辑选中的电缆"""
        current = self.cable_list.currentItem()
        if not current:
            QMessageBox.information(self, "No Selection", "Please select a cable to edit.")
            return

        # 解析当前选中的电缆
        text = current.text()
        cable_id = text.split(' : ')[0]

        # 找到电缆数据
        for i, (cid, start, end) in enumerate(self.cables):
            if cid == cable_id:
                dialog = CableConfigDialog(self, self.router.width, self.router.height, edit_mode=True)
                dialog.set_cable_data({'id': cid, 'start': start, 'end': end})

                if dialog.exec() == QDialog.DialogCode.Accepted:
                    data = dialog.get_cable_data()
                    self.cables[i] = (data['id'], data['start'], data['end'])
                    self.update_cable_list()
                    self.status_label.setText(f"Updated cable: {data['id']}")
                break

    def remove_cable(self):
        """删除选中的电缆"""
        current = self.cable_list.currentItem()
        if not current:
            QMessageBox.information(self, "No Selection", "Please select a cable to remove.")
            return

        text = current.text()
        cable_id = text.split(' : ')[0]

        reply = QMessageBox.question(self, "Confirm Remove",
                                     f"Remove cable '{cable_id}'?",
                                     QMessageBox.StandardButton.Yes |
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.cables = [c for c in self.cables if c[0] != cable_id]
            self.update_cable_list()

            # 从路径中移除
            if cable_id in self.router.cable_paths:
                del self.router.cable_paths[cable_id]

            self.visualizer.update()
            self.status_label.setText(f"Removed cable: {cable_id}")

    # def recalculate_paths(self):
    #     """重新计算路径"""
    #     self.status_label.setText("Calculating paths...")
    #     self.status_label.repaint()
    #     QApplication.processEvents()
    #
    #     # 运行路径规划
    #     success = self.router.run(self.cables)
    #     if success:
    #         self.status_label.setText("Paths calculated successfully!")
    #     else:
    #         self.status_label.setText("Warning: Some paths could not be found!")
    #     # 刷新显示
    #     self.visualizer.update()

    def recalculate_paths(self):
        self.status_label.setText("Calculating paths...")

        self.thread = SolverThread(self.router, self.cables)
        self.thread.finished_signal.connect(self.on_solver_finished)
        self.thread.start()

    def on_solver_finished(self, success):
        if success:
            self.status_label.setText("Paths calculated successfully!")
        else:
            self.status_label.setText("Warning: Some paths could not be found!")

        self.visualizer.update()

    def clear_obstacles(self):
        """清除所有障碍物（除了边界）"""
        reply = QMessageBox.question(self, "Confirm Clear",
                                     "Clear all obstacles?",
                                     QMessageBox.StandardButton.Yes |
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
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

    def clear_paths(self):
        """清除所有路径"""
        reply = QMessageBox.question(self, "Confirm Clear",
                                     "Clear all paths?",
                                     QMessageBox.StandardButton.Yes |
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.router.cable_paths = {}
            self.visualizer.update()
            self.status_label.setText("Paths cleared.")

    def save_configuration(self):
        """保存配置到JSON文件"""
        # 如果已经有当前文件，直接保存
        if self.current_file and os.path.exists(self.current_file):
            filename = self.current_file
        else:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Configuration", "",
                "JSON Files (*.json);;All Files (*)"
            )
            if not filename:
                return
            if not filename.endswith('.json'):
                filename += '.json'
            self.current_file = filename

        try:
            self.router.save_to_json(filename)
            self.file_label.setText(f"File: {os.path.basename(filename)}")
            self.status_label.setText(f"Configuration saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving file: {str(e)}")

    def load_configuration(self):
        """从JSON文件加载配置"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not filename:
            return

        try:
            # 保存旧的电缆数据用于验证
            old_cables = self.cables.copy()

            # 加载配置
            new_cables = self.router.load_from_json(filename)
            # 2. 更新主窗口的电缆列表
            # 注意：不要直接 self.cables = new_cables，最好清空后 extend
            self.cables.clear()
            for c in new_cables:
                # 确保坐标是 tuple 类型 (JSON 加载出来可能是 list)
                self.cables.append((c[0], tuple(c[1]), tuple(c[2])))

            # 3. 关键：更新可视化组件的引用
            self.visualizer.cables_ref = self.cables
            # 从路径重建电缆列表
            # self.cables = []
            # for cid, path in self.router.cable_paths.items():
            #     if path:
            #         start = path[0][0]
            #         end = path[-1][1]
            #         self.cables.append((cid, start, end))

            self.update_cable_list()
            self.obstacle_count_label.setText(str(len(self.router.obstacles)))
            self.visualizer.update()

            self.current_file = filename
            self.file_label.setText(f"File: {os.path.basename(filename)}")
            self.status_label.setText(f"Configuration loaded from {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading file: {str(e)}")

    def export_npz(self):
        """导出为NPZ格式"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export as NPZ", "",
            "NPZ Files (*.npz);;All Files (*)"
        )
        if not filename:
            return

        if not filename.endswith('.npz'):
            filename += '.npz'

        try:
            self.router.save_to_npz(filename)
            self.status_label.setText(f"Exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting file: {str(e)}")


# --- 运行测试 ---
if __name__ == "__main__":
    width = 10
    height = 10

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
    obstacles.add((6, 3))
    obstacles.add((7, 4))

    # 创建路由器
    router = HarnessRouterGrid(
        width,
        height,
        obstacles,
        wl=0.5,
        wb=1.2
    )

    # 定义电缆起点终点
    cables = [
        ('Cable_Red', (2, 2), (7, 7)),
        ('Cable_Blue', (3, 3), (6, 6)),
        ('Cable_Green', (4, 4), (3, 8))
    ]

    # 创建PyQt6应用
    app = QApplication(sys.argv)
    window = MainWindow(router, cables)
    window.show()
    sys.exit(app.exec())