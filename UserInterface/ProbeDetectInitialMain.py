# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox,QInputDialog,qApp,QDialog
from PyQt5.QtGui import QImage, QPixmap,QPen,QBrush,QPainter
from PyQt5.QtCore import Qt,QRect,pyqtSignal
from UserInterface.ProbeDetectInitial import Ui_ProbeDetectInitial
from UserInterface.ViewTool import IMG_WIN
import cv2
import socket
import struct
import ctypes
import datetime
import threading
import time
import sys, os


class ProbeDetectInittalServer(QMainWindow, Ui_ProbeDetectInitial):
    def __init__(self, parent=None):
        super(ProbeDetectInittalServer, self).__init__(parent)
        self.save = 0
        self.IP_address = "192.168.0.121"
        self.setupUi(self)
        self.graphic = IMG_WIN(self.ImageShow)
        filePath = './../sources/KR_A/rack_front_0.bmp'
        img = cv2.imread(filePath)
        self.graphic.addScenes(img)

        self.Point1.clicked.connect(self.click_point1)
        self.Point2.clicked.connect(self.click_point2)
        self.template1.clicked.connect(self.template1_event)
        self.template2.clicked.connect(self.template2_event)
        self.template3.clicked.connect(self.template3_event)
        self.template4.clicked.connect(self.template4_event)
        self.template5.clicked.connect(self.template5_event)
        self.template6.clicked.connect(self.template6_event)
        self.template7.clicked.connect(self.template7_event)
        self.template8.clicked.connect(self.template8_event)
        self.template9.clicked.connect(self.template9_event)
        self.template10.clicked.connect(self.template10_event)
        self.template11.clicked.connect(self.template11_event)
        self.template12.clicked.connect(self.template12_event)
        self.template13.clicked.connect(self.template13_event)
        self.template14.clicked.connect(self.template14_event)
        self.template15.clicked.connect(self.template15_event)

        self.IP.textChanged.connect(self.IPTextChange)
        # 子线程消息显示,信号槽机制
        self.signal_show_point1.connect(self.show_Point1)
        self.signal_show_point2.connect(self.show_Point2)
        #self.showMaximized()
        #显示处理后的图像,这里也得用线程处理
        #image = QPixmap("./display_image.png").scaled(self.ImageShow.width(),self.ImageShow.height())
        #self.ImageShow.setPixmap(image)
        #连接部分

    def IPTextChange(self):
        self.IP_address = self.IP.toPlainText()
    def click_point1(self):
        point1 = self.graphic.getPoint()
        if point1 is not None:
            msg = '(' + str(int(point1[0])) + ',' + str(int(point1[1])) + ')'
            self.signal_show_point1.emit(msg)

    def click_point2(self):
        point2 = self.graphic.getPoint()
        if point2 is not None:
            msg = '(' + str(int(point2[0])) + ',' + str(int(point2[1])) + ')'
            self.signal_show_point2.emit(msg)

    def show_Point1(self, msg):

        self.Point1Show.clear()
        self.Point1Show.setText(msg)

    def show_Point2(self, msg):
        #msg = '(' + str(point2[0]) + ',' + str(point2[1]) + ')'
        self.Point2Show.clear()
        self.Point2Show.setText(msg)

    def template1_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 1
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template2_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 2
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template3_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 3
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template4_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 4
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template5_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 5
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template6_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 6
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template7_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 7
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template8_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 8
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template9_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 9
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template10_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 10
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template11_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 11
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template12_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 12
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template13_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 13
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template14_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 14
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)

    def template15_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 15
        if self.save == 1:
            self.graphic.draw_rect = True
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            img = self.graphic.org
            x1 = int(self.graphic.pixmap_rect.x())
            x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
            y1 = int(self.graphic.pixmap_rect.y())
            y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
            img = img[y1:y2, x1:x2, :]
            filename = './../sources/KR_A/temp/new_2/temp_' + str(idx-1) + '.bmp'
            cv2.imwrite(filename, img)
            for i, button in enumerate(self.buttonlist):
                if i == idx - 1:
                    continue
                else:
                    button.setEnabled(True)
if __name__ == "__main__":
    #固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    #初始化
    myWin = ProbeDetectInittalServer()
    #将窗口控件显示在屏幕上
    myWin.show()
    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
