from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QBrush, QColor,QPainter
from PyQt5.QtWidgets import QWidget
import cv2
from PyQt5.QtWidgets import QGraphicsRectItem

#self.graphic = IMG_WIN(graphicsView)
#img = cv2.imread(filePath)
#self.graphic.addScenes(img)


class IMG_WIN(QWidget):
    def __init__(self, graphicsView):
        super().__init__()
        self.point = None
        self.draw_rect = False
        self.left_top = None
        self.right_bottom = None
        self.rect = QGraphicsRectItem()
        self.scene_rect = None
        self.pixmap_rect = None
        self.graphicsView = graphicsView

        self.graphicsView.setStyleSheet("padding: 0px; border: 0px;")  # 内边距和边界去除
        self.scene = QtWidgets.QGraphicsScene(self)
        self.graphicsView.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)  # 改变对齐方式

        self.graphicsView.setSceneRect(0, 0, self.graphicsView.viewport().width(),
                                       self.graphicsView.height())  # 设置图形场景大小和图形视图大小一致
        self.graphicsView.setScene(self.scene)

        self.scene.mousePressEvent = self.scene_MousePressEvent  # 接管图形场景的鼠标点击事件
        self.scene.mouseReleaseEvent = self.scene_mouseReleaseEvent
        self.scene.mouseMoveEvent = self.scene_mouseMoveEvent # 接管图形场景的鼠标移动事件
        self.scene.wheelEvent = self.scene_wheelEvent			# 接管图形场景的滑轮事件


        self.w_ratio = 1
        self.h_ratio = 1
        self.ratio = 1  # 缩放初始比例
        self.zoom_step = 0.1  # 缩放步长
        self.zoom_max = 2  # 缩放最大值
        self.zoom_min = 0.01  # 缩放最小值
        self.pixmapItem = None

    def getPoint(self):
        tem = self.point
        self.point = None
        return tem

    def addScenes(self,img):  # 绘制图形
        self.org = img
        self.ori_wide = img.shape[1]
        self.ori_height = img.shape[0]
        if self.pixmapItem != None:
            originX = self.pixmapItem.x()
            originY = self.pixmapItem.y()
        else:
            originX, originY = 0, 0  # 坐标基点

        self.scene.clear() # 清除当前图元
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
        self.pixmap = QtGui.QPixmap(
            QtGui.QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3,
                         QtGui.QImage.Format_RGB888))  # 转化为qlbel格式

        self.pixmapItem = self.scene.addPixmap(self.pixmap)
        self.pixmapItem.setScale(self.ratio)  # 缩放
        self.pixmapItem.setPos(originX, originY)

    def scene_MousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:  # 右键按下
            if self.draw_rect:
                self.left_top = event.scenePos()
            self.right_preMousePosition = event.scenePos()  # 获取鼠标当前位置
            image_xx = self.pixmapItem.pos().x()
            image_yy = self.pixmapItem.pos().y()
            cur_xx = (self.right_preMousePosition.x() - image_xx) / self.pixmapItem.scale()
            cur_yy = (self.right_preMousePosition.y() - image_yy) / self.pixmapItem.scale()
            self.point = (cur_xx, cur_yy)
        if event.button() == QtCore.Qt.LeftButton:  # 左键按下
            # print("鼠标左键单击")  # 响应测试语句
            # print(event.scenePos())
            self.preMousePosition = event.scenePos()  # 获取鼠标当前位置


    def scene_mouseReleaseEvent(self, event):

        if event.button() == QtCore.Qt.RightButton:
           return
                #self.pixmapItem.boundingRect(QRectF(0, 0, 100, 100))




    def scene_mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.RightButton:

            if self.draw_rect:
                self.right_bottom = event.scenePos()
                x = self.left_top.x()
                y = self.left_top.y()
                w = self.right_bottom.x() - x
                h = self.right_bottom.y() - y
                self.scene_rect = QRectF(x, y, w, h)
                self.rect.setRect(self.scene_rect)
                self.scene.addItem(self.rect)
                self.rect.setPen(QColor(255, 0, 0))
                #self.rect.setBrush(QtCore.Qt.CrossPattern)
                #self.rect.setBrush(QBrush())

                self.rect.show()
                self.pixmap_rect = self.pixmapItem.mapRectFromScene(self.scene_rect)
                #print(self.pixmap_rect.x(),self.pixmap_rect.y(),self.pixmap_rect.width(),self.pixmap_rect.height())
                #self.pixmapItem.boundingRect(QRectF(0, 0, 100, 100))

            #self.pixmapItem.boundingRect(QRectF(left_top_x, left_top_y, 100, 100))
            #print(left_top_x, left_top_y)
            # rect = QGraphicsRectItem(left_top_y)
            # pen = rect.pen()
            # pen.setWidth(5)
            # pen.setColor()
            # rect.setPen(pen)
            # rect.setBrush(QBrush(QColor(0, 160, 230)))
            #
            # rect.setRect(QRectF(50, 50, 100, 100));
            # self.scene.addItem(rect);

        if event.buttons() == QtCore.Qt.LeftButton:
            # print("左键移动")  # 响应测试语句
            self.MouseMove = event.scenePos() - self.preMousePosition  # 鼠标当前位置-先前位置=单次偏移量
            self.preMousePosition = event.scenePos()  # 更新当前鼠标在窗口上的位置，下次移动用
            self.pixmapItem.setPos(self.pixmapItem.pos() + self.MouseMove)  # 更新图元位置
            #print(self.pixmapItem.pos())
        # 定义滚轮方法。当鼠标在图元范围之外，以图元中心为缩放原点；当鼠标在图元之中，以鼠标悬停位置为缩放中心

    def scene_wheelEvent(self, event):
        angle = event.delta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        if angle > 0:
            # print("滚轮上滚")
            self.ratio += self.zoom_step  # 缩放比例自加
            if self.ratio > self.zoom_max:
                self.ratio = self.zoom_max
            else:
                #print(self.pixmap.size())
                w = self.pixmap.size().width() * (self.ratio - self.zoom_step)
                h = self.pixmap.size().height() * (self.ratio - self.zoom_step)
                x1 = self.pixmapItem.pos().x()  # 图元左位置
                x2 = self.pixmapItem.pos().x() + w  # 图元右位置
                y1 = self.pixmapItem.pos().y()  # 图元上位置
                y2 = self.pixmapItem.pos().y() + h  # 图元下位置
                if event.scenePos().x() > x1 and event.scenePos().x() < x2 \
                        and event.scenePos().y() > y1 and event.scenePos().y() < y2:  # 判断鼠标悬停位置是否在图元中
                    print('在内部')
                    self.pixmapItem.setScale(self.ratio)  # 缩放
                    a1 = event.scenePos() - self.pixmapItem.pos()  # 鼠标与图元左上角的差值
                    a2 = self.ratio/(self.ratio - self.zoom_step)-1    # 对应比例
                    delta = a1 * a2
                    self.pixmapItem.setPos(self.pixmapItem.pos() - delta)
                    # ----------------------------分维度计算偏移量-----------------------------
                    # delta_x = a1.x()*a2
                    # delta_y = a1.y()*a2
                    # self.pixmapItem.setPos(self.pixmapItem.pos().x() - delta_x,
                    #                        self.pixmapItem.pos().y() - delta_y)  # 图元偏移
                    # -------------------------------------------------------------------------

                else:
                    print('在外部')  # 以图元中心缩放
                    self.pixmapItem.setScale(self.ratio)  # 缩放
                    delta_x = (self.pixmap.size().width() * self.zoom_step) / 2  # 图元偏移量
                    delta_y = (self.pixmap.size().height() * self.zoom_step) / 2
                    self.pixmapItem.setPos(self.pixmapItem.pos().x() - delta_x,
                                           self.pixmapItem.pos().y() - delta_y)  # 图元偏移
        else:
            # print("滚轮下滚")
            self.ratio -= self.zoom_step
            if self.ratio < self.zoom_min:
                self.ratio = self.zoom_min
            else:
                w = self.pixmap.size().width() * (self.ratio + self.zoom_step)
                h = self.pixmap.size().height() * (self.ratio + self.zoom_step)
                x1 = self.pixmapItem.pos().x()
                x2 = self.pixmapItem.pos().x() + w
                y1 = self.pixmapItem.pos().y()
                y2 = self.pixmapItem.pos().y() + h
                # print(x1, x2, y1, y2)
                if event.scenePos().x() > x1 and event.scenePos().x() < x2 \
                        and event.scenePos().y() > y1 and event.scenePos().y() < y2:
                    # print('在内部')
                    self.pixmapItem.setScale(self.ratio)  # 缩放
                    a1 = event.scenePos() - self.pixmapItem.pos()  # 鼠标与图元左上角的差值
                    a2=self.ratio/(self.ratio+ self.zoom_step)-1    # 对应比例
                    delta = a1 * a2
                    self.pixmapItem.setPos(self.pixmapItem.pos() - delta)
                    # ----------------------------分维度计算偏移量-----------------------------
                    # delta_x = a1.x()*a2
                    # delta_y = a1.y()*a2
                    # self.pixmapItem.setPos(self.pixmapItem.pos().x() - delta_x,
                    #                        self.pixmapItem.pos().y() - delta_y)  # 图元偏移
                    # -------------------------------------------------------------------------
                else:
                    # print('在外部')
                    self.pixmapItem.setScale(self.ratio)
                    delta_x = (self.pixmap.size().width() * self.zoom_step) / 2
                    delta_y = (self.pixmap.size().height() * self.zoom_step) / 2
                    self.pixmapItem.setPos(self.pixmapItem.pos().x() + delta_x, self.pixmapItem.pos().y() + delta_y)


class IMG_WIN2(QWidget):

    def __init__(self, graphicsView):
        super().__init__()

        self.draw_rect = False
        self.left_top = None
        self.right_bottom = None
        self.rect = QGraphicsRectItem()
        self.scene_rect = None
        self.pixmap_rect = None
        self.graphicsView = graphicsView

        self.graphicsView.setStyleSheet("padding: 0px; border: 0px;")  # 内边距和边界去除
        self.scene = QtWidgets.QGraphicsScene(self)
        self.graphicsView.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)  # 改变对齐方式

        self.graphicsView.setSceneRect(0, 0, self.graphicsView.viewport().width(),
                                       self.graphicsView.height())  # 设置图形场景大小和图形视图大小一致
        self.graphicsView.setScene(self.scene)

        self.scene.mousePressEvent = self.scene_MousePressEvent  # 接管图形场景的鼠标点击事件
        self.scene.mouseMoveEvent = self.scene_mouseMoveEvent # 接管图形场景的鼠标移动事件
        self.scene.wheelEvent = self.scene_wheelEvent			# 接管图形场景的滑轮事件


        self.w_ratio = 1
        self.h_ratio = 1
        self.ratio = 1  # 缩放初始比例
        self.zoom_step = 0.1  # 缩放步长
        self.zoom_max = 2  # 缩放最大值
        self.zoom_min = 0.01  # 缩放最小值
        self.pixmapItem = None


    def addScenes(self,img):  # 绘制图形
        self.org = img
        self.ori_wide = img.shape[1]
        self.ori_height = img.shape[0]
        if self.pixmapItem != None:
            originX = self.pixmapItem.x()
            originY = self.pixmapItem.y()
        else:
            originX, originY = 0, 0  # 坐标基点

        self.scene.clear() # 清除当前图元
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
        self.pixmap = QtGui.QPixmap(
            QtGui.QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3,
                         QtGui.QImage.Format_RGB888))  # 转化为qlbel格式

        self.pixmapItem = self.scene.addPixmap(self.pixmap)
        self.pixmapItem.setScale(self.ratio)  # 缩放
        self.pixmapItem.setPos(originX, originY)

    def scene_MousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:  # 右键按下
            if self.draw_rect:
                self.left_top = event.scenePos()

        if event.button() == QtCore.Qt.LeftButton:  # 左键按下
            self.preMousePosition = event.scenePos()  # 获取鼠标当前位置

    def scene_mouseMoveEvent(self, event):

        if event.buttons() == QtCore.Qt.RightButton:

            if self.draw_rect:
                self.right_bottom = event.scenePos()
                x = self.left_top.x()
                y = self.left_top.y()
                w = self.right_bottom.x() - x
                h = self.right_bottom.y() - y
                self.scene_rect = QRectF(x, y, w, h)
                self.rect.setRect(self.scene_rect)
                self.scene.addItem(self.rect)
                self.rect.setPen(QColor(255, 0, 0))
                self.rect.show()
                self.pixmap_rect = self.pixmapItem.mapRectFromScene(self.scene_rect)

        if event.buttons() == QtCore.Qt.LeftButton:
            # print("左键移动")  # 响应测试语句
            self.MouseMove = event.scenePos() - self.preMousePosition  # 鼠标当前位置-先前位置=单次偏移量
            self.preMousePosition = event.scenePos()  # 更新当前鼠标在窗口上的位置，下次移动用
            self.pixmapItem.setPos(self.pixmapItem.pos() + self.MouseMove)  # 更新图元位置
        # 定义滚轮方法。当鼠标在图元范围之外，以图元中心为缩放原点；当鼠标在图元之中，以鼠标悬停位置为缩放中心

    def scene_wheelEvent(self, event):
        angle = event.delta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        if angle > 0:
            # print("滚轮上滚")
            self.ratio += self.zoom_step  # 缩放比例自加
            if self.ratio > self.zoom_max:
                self.ratio = self.zoom_max
            else:
                #print(self.pixmap.size())
                w = self.pixmap.size().width() * (self.ratio - self.zoom_step)
                h = self.pixmap.size().height() * (self.ratio - self.zoom_step)
                x1 = self.pixmapItem.pos().x()  # 图元左位置
                x2 = self.pixmapItem.pos().x() + w  # 图元右位置
                y1 = self.pixmapItem.pos().y()  # 图元上位置
                y2 = self.pixmapItem.pos().y() + h  # 图元下位置
                if event.scenePos().x() > x1 and event.scenePos().x() < x2 \
                        and event.scenePos().y() > y1 and event.scenePos().y() < y2:  # 判断鼠标悬停位置是否在图元中
                    self.pixmapItem.setScale(self.ratio)  # 缩放
                    a1 = event.scenePos() - self.pixmapItem.pos()  # 鼠标与图元左上角的差值
                    a2 = self.ratio/(self.ratio - self.zoom_step)-1    # 对应比例
                    delta = a1 * a2
                    self.pixmapItem.setPos(self.pixmapItem.pos() - delta)
                    # ----------------------------分维度计算偏移量-----------------------------
                    # delta_x = a1.x()*a2
                    # delta_y = a1.y()*a2
                    # self.pixmapItem.setPos(self.pixmapItem.pos().x() - delta_x,
                    #                        self.pixmapItem.pos().y() - delta_y)  # 图元偏移
                    # -------------------------------------------------------------------------

                else:
                    self.pixmapItem.setScale(self.ratio)  # 缩放
                    delta_x = (self.pixmap.size().width() * self.zoom_step) / 2  # 图元偏移量
                    delta_y = (self.pixmap.size().height() * self.zoom_step) / 2
                    self.pixmapItem.setPos(self.pixmapItem.pos().x() - delta_x,
                                           self.pixmapItem.pos().y() - delta_y)  # 图元偏移
        else:
            # print("滚轮下滚")
            self.ratio -= self.zoom_step
            if self.ratio < self.zoom_min:
                self.ratio = self.zoom_min
            else:
                w = self.pixmap.size().width() * (self.ratio + self.zoom_step)
                h = self.pixmap.size().height() * (self.ratio + self.zoom_step)
                x1 = self.pixmapItem.pos().x()
                x2 = self.pixmapItem.pos().x() + w
                y1 = self.pixmapItem.pos().y()
                y2 = self.pixmapItem.pos().y() + h
                # print(x1, x2, y1, y2)
                if event.scenePos().x() > x1 and event.scenePos().x() < x2 \
                        and event.scenePos().y() > y1 and event.scenePos().y() < y2:
                    # print('在内部')
                    self.pixmapItem.setScale(self.ratio)  # 缩放
                    a1 = event.scenePos() - self.pixmapItem.pos()  # 鼠标与图元左上角的差值
                    a2=self.ratio/(self.ratio+ self.zoom_step)-1    # 对应比例
                    delta = a1 * a2
                    self.pixmapItem.setPos(self.pixmapItem.pos() - delta)
                    # ----------------------------分维度计算偏移量-----------------------------
                    # delta_x = a1.x()*a2
                    # delta_y = a1.y()*a2
                    # self.pixmapItem.setPos(self.pixmapItem.pos().x() - delta_x,
                    #                        self.pixmapItem.pos().y() - delta_y)  # 图元偏移
                    # -------------------------------------------------------------------------
                else:
                    # print('在外部')
                    self.pixmapItem.setScale(self.ratio)
                    delta_x = (self.pixmap.size().width() * self.zoom_step) / 2
                    delta_y = (self.pixmap.size().height() * self.zoom_step) / 2
                    self.pixmapItem.setPos(self.pixmapItem.pos().x() + delta_x, self.pixmapItem.pos().y() + delta_y)
