from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QFileDialog, QDockWidget, QTreeWidget,
    QTreeWidgetItem, QMenuBar, QGraphicsPixmapItem, QMessageBox,
    QColorDialog, QInputDialog, QDialog, QVBoxLayout, QToolBar
)
from PyQt6.QtGui import QPixmap, QBrush, QAction, QPainter, QColor, QImage
from PyQt6.QtCore import Qt, QPointF
from fisheye_calibrator import FisheyeCalibrator
import cv2
import numpy as np
from PyQt6.QtWidgets import QMenu

class ImageAnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像标注工具")
        self.setGeometry(100, 100, 800, 600)

        self.scene = QGraphicsScene()
        self.view = GraphicsView(self.scene)
        self.setCentralWidget(self.view)

        self._initialize()

        self._create_toolbar()
        self._create_coordinate_list()
    
    def _initialize(self):
        """初始化变量和状态"""
        self.current_image_path = None
        self.image_size = None
        self.image = None
        self.classes = []
        self.selected_class_index = 0

    def _create_initial_class(self):
        """创建初始默认分类"""
        initial_color = QColor.fromHsv(0, 255, 255)  # 红色
        self.classes.append({
            'name': 'Line 1',
            'color': initial_color,
            'points': [],
            'dots': []
        })
        self.selected_class_index = 0

    def _create_toolbar(self):
        """创建带有常用按钮的工具栏"""
        toolbar = QToolBar("主工具栏")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        # 打开文件按钮
        open_action = QAction("打开", self)
        open_action.triggered.connect(self._open_image)
        toolbar.addAction(open_action)

        # 新建分类按钮
        new_class_action = QAction("新建分类", self)
        new_class_action.triggered.connect(self._create_new_class)
        toolbar.addAction(new_class_action)

        # 分隔线
        toolbar.addSeparator()

        # 处理按钮
        process_action = QAction("去畸变", self)
        process_action.triggered.connect(self.on_process_triggered)
        toolbar.addAction(process_action)

        # 设置工具提示
        open_action.setToolTip("打开图像文件")
        new_class_action.setToolTip("创建新分类")
        process_action.setToolTip("执行去畸变处理")

    def _generate_next_color(self):
        """生成下一个分类颜色"""
        hue = (len(self.classes) * 50) % 360
        return QColor.fromHsv(hue, 255, 255)

    def _create_new_class(self):
        """创建新分类（自动分配颜色）"""
        color = self._generate_next_color()
        name = "Line {}".format(len(self.classes) + 1)
        new_class = {
            'name': name,
            'color': color,
            'points': [],
            'dots': []
        }
        self.classes.append(new_class)
        self.selected_class_index = len(self.classes) - 1
        self._update_coordinate_tree()

    def _create_coordinate_list(self):
        """创建分类坐标树"""
        self.coord_tree = QTreeWidget()
        self.coord_tree.setHeaderLabels(["类别", "坐标"])
        dock_widget = QDockWidget("标注列表", self)
        dock_widget.setWidget(self.coord_tree)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_widget)
        self._update_coordinate_tree()

        self.coord_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.coord_tree.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, position):
        item = self.coord_tree.itemAt(position)
        if not item:
            return

        menu = QMenu(self)

        parent = item.parent()
        if parent is None:
            delete_action = menu.addAction("删除该分类")
            action = menu.exec(self.coord_tree.mapToGlobal(position))
            if action == delete_action:
                index = self.coord_tree.indexOfTopLevelItem(item)
                if index >= 0:
                    reply = QMessageBox.question(self, "确认", f"确定删除分类 '{self.classes[index]['name']}'？",
                                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.Yes:
                        self._delete_class(index)
        else:
            class_index = self.coord_tree.indexOfTopLevelItem(parent)
            point_index = parent.indexOfChild(item)
            delete_action = menu.addAction("删除该点")
            action = menu.exec(self.coord_tree.mapToGlobal(position))
            if action == delete_action:
                self._delete_point(class_index, point_index)

    def _update_coordinate_tree(self):
        """更新坐标树显示"""
        self.coord_tree.clear()
        for cls in self.classes:
            class_item = QTreeWidgetItem([cls['name'], ""])
            class_item.setForeground(0, QBrush(cls['color']))
            for point in cls['points']:
                coord_item = QTreeWidgetItem(["", f"({point[0]:.2f}, {point[1]:.2f})"])
                coord_item.setForeground(1, QBrush(cls['color']))
                class_item.addChild(coord_item)
            self.coord_tree.addTopLevelItem(class_item)
        self.coord_tree.expandAll()

    def _delete_class(self, class_index):
        cls = self.classes[class_index]
        for dot in cls['dots']:
            self.scene.removeItem(dot)
        del self.classes[class_index]

        if len(self.classes) == 0:
            initial_color = QColor.fromHsv(0, 255, 255)  
            self.classes.append({
                'name': 'Line 1',
                'color': initial_color,
                'points': [],
                'dots': []
            })
            self.selected_class_index = 0
        else:
            self.selected_class_index = min(class_index, len(self.classes) - 1)

        self._update_coordinate_tree()

    def _delete_point(self, class_index, point_index):
        cls = self.classes[class_index]
        dot_item = cls['dots'].pop(point_index)
        if dot_item:
            self.scene.removeItem(dot_item)
        cls['points'].pop(point_index)
        self._update_coordinate_tree()

    def _open_image(self):
        """打开图像文件"""
        self._initialize()
        self._create_initial_class()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)"
        )
        if filename:
            self.current_image_path = filename
            self._load_image(filename)

    def _load_image(self, filename):
        """加载图像并显示"""
        self.image = cv2.imread(filename)
        if self.image is None:
            QMessageBox.critical(self, "错误", "无法加载图像")
            return
        self.image_size = self.image.shape[:2][::-1]  # (width, height)

        for cls in self.classes:
            cls['points'].clear()
            for dot in cls['dots']:
                self.scene.removeItem(dot)
            cls['dots'].clear()
        self.coord_tree.clear()

        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        if pixmap.isNull():
            return
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(pixmap.rect().toRectF())
        self.view.fit_to_view()

    def add_point(self, pos):
        """添加标注点"""
        current_class = self.classes[self.selected_class_index]
        dot = QGraphicsEllipseItem(-3, -3, 6, 6)
        dot.setPos(pos)
        dot.setBrush(QBrush(current_class['color']))
        self.scene.addItem(dot)
        current_class['points'].append((pos.x(), pos.y()))
        current_class['dots'].append(dot)
        self._update_coordinate_tree()

    def on_process_triggered(self):
        """处理去畸变操作"""
        if not self.current_image_path:
            QMessageBox.warning(self, "错误", "请先加载图像")
            return
        
        valid = any(len(cls['points']) >= 2 for cls in self.classes)
        if not valid:
            QMessageBox.warning(self, "错误", "至少需要一个类别包含两个点")
            return

        calibrator = FisheyeCalibrator(self.classes, self.image_size)
        calibrator.optimize_step1()
        calibrator.optimize_step2()
        calibrator.optimize_step3()
        K, D = calibrator.K, calibrator.D
        print(f"优化后的内参矩阵K:\n{K}")
        print(f"优化后的畸变系数D:\n{D}")
        self.show_comparison(K, D)

    def show_comparison(self, K, D):
        """显示原图与去畸变后的对比图"""
        try:
            undistorted_img = cv2.fisheye.undistortImage(self.image, K, D, Knew=K)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"去畸变失败: {str(e)}")
            return

        # 拼接原图和去畸变后的图像
        comparison_img = np.hstack((self.image, undistorted_img))
        cv2.putText(comparison_img, 'Original', (10,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(comparison_img, 'Undistorted', (self.image.shape[1]+10,30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        height, width = comparison_img.shape[:2]
        bytes_per_line = 3 * width
        rgb_image = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        if qimage.isNull():
            QMessageBox.critical(self, "错误", "无法生成对比图像")
            return

        pixmap = QPixmap.fromImage(qimage)

        dialog = QDialog(self)
        dialog.setWindowTitle("去畸变对比 - 左: 原始图像, 右: 校正后图像")
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        scene.addPixmap(pixmap)

        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        layout = QVBoxLayout(dialog)
        layout.addWidget(view)
        dialog.setLayout(layout)

        screen_geo = QApplication.primaryScreen().availableGeometry()
        max_width = screen_geo.width() - 100
        max_height = screen_geo.height() - 100
        img_width = pixmap.width()
        img_height = pixmap.height()

        if img_width > max_width or img_height > max_height:
            dialog.resize(max_width, max_height)
        else:
            dialog.resize(img_width + 20, img_height + 20)

        dialog.exec()

class GraphicsView(QGraphicsView):
    ZOOM_FACTOR = 1.15
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self.zoom_level = 0
        self.max_zoom = 10
        self.min_zoom = -10

        self._pan = False
        self._pan_start = QPointF()
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def wheelEvent(self, event):
        """实现以鼠标为中心的精确缩放"""
        old_pos = self.mapToScene(event.position().toPoint())

        if event.angleDelta().y() > 0:
            zoom = GraphicsView.ZOOM_FACTOR
            self.zoom_level += 1
        else:
            zoom = 1 / GraphicsView.ZOOM_FACTOR
            self.zoom_level -= 1

        if self.zoom_level > self.max_zoom or self.zoom_level < self.min_zoom:
            self.zoom_level = max(min(self.zoom_level, self.max_zoom), self.min_zoom)
            return

        self.scale(zoom, zoom)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._pan = True
            self._pan_start = self.mapToScene(event.position().toPoint())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._pan = False
            self.setCursor(Qt.CursorShape.CrossCursor)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """处理拖拽平移"""
        if self._pan:
            new_pos = self.mapToScene(event.position().toPoint())
            delta = new_pos - self._pan_start
            
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self._pan_start = self.mapToScene(event.position().toPoint())
        super().mouseMoveEvent(event)

    def fit_to_view(self):
        """重置视图时保持缩放级别归零"""
        self.zoom_level = 0
        if self.scene().items():
            self.fitInView(self.scene().items()[0], Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            image_item = next((item for item in self.scene().items() if isinstance(item, QGraphicsPixmapItem)), None)
            
            if image_item:
                scene_pos = self.mapToScene(event.position().toPoint())
                item_pos = image_item.mapFromScene(scene_pos)
                
                pixmap = image_item.pixmap()
                if 0 <= item_pos.x() < pixmap.width() and 0 <= item_pos.y() < pixmap.height():
                    self.parent().add_point(item_pos)
            else:
                QMessageBox.warning(self, "错误", "未找到图像项")
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event):
        """处理键盘事件"""
        if event.key() == Qt.Key.Key_N:
            self.parent()._create_new_class()
        elif event.key() == Qt.Key.Key_P:
            self.parent().on_process_triggered()
        elif event.key() == Qt.Key.Key_Escape:
            self.parent().close()
        elif event.key() == Qt.Key.Key_O:
            self.parent()._open_image()
        else:
            super().keyPressEvent(event)