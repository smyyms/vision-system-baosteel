# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox,QInputDialog,qApp,QDialog
from PyQt5.QtGui import QImage, QPixmap,QPen,QBrush,QPainter
from PyQt5.QtCore import Qt,QRect,pyqtSignal
from UserInterface.TemperatureMeasureInitial import Ui_TemperatureMeasureInitial
from UserInterface.ViewTool import IMG_WIN2
import cv2
import socket
import struct
import ctypes
import datetime
import threading
import time
import sys, os


class TemperatureMeasureInitialServer(QMainWindow, Ui_TemperatureMeasureInitial):
    def __init__(self, parent=None):
        super(TemperatureMeasureInitialServer, self).__init__(parent)
        self.save = 0
        self.secure_ROI = []
        self.roi_save = 0
        self.IP_address = "192.168.0.122"
        self.setupUi(self)
        self.graphic = IMG_WIN2(self.ImageShow)
        filePath = './../sources/KR_C/test.bmp'
        img = cv2.imread(filePath)
        self.graphic.addScenes(img)

        self.ROI.clicked.connect(self.click_ROI)
        self.template1.clicked.connect(self.template1_event)

        self.IP.textChanged.connect(self.IPTextChange)
        # 子线程消息显示,信号槽机制
        self.signal_show_LT.connect(self.show_LT)
        self.signal_show_RB.connect(self.show_RB)


    def IPTextChange(self):

        #self.IP_address = self.IP.Text()
        print(self.IP_address)

    def click_ROI(self):
        self.roi_save = self.roi_save + 1
        self.roi_save = self.roi_save % 2

        if self.roi_save == 1:
            self.graphic.draw_rect = True
            self.template1.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            rect = self.graphic.pixmap_rect

            if rect is not None:

                point1 = [int(rect.x()), int(rect.y())]
                point2 = [int(rect.x() + rect.width()), int(rect.y() + rect.height())]
                self.secure_ROI = [int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())]
                msg1 = '(' + str(point1[0]) + ',' + str(point1[1]) + ')'
                msg2 = '(' + str(point2[0]) + ',' + str(point2[1]) + ')'
                self.signal_show_LT.emit(msg1)
                self.signal_show_RB.emit(msg2)

                self.graphic.pixmap_rect = None
            self.template1.setEnabled(True)

    def show_LT(self, msg):

        self.LT_show.clear()
        self.LT_show.setText(msg)

    def show_RB(self, msg):

        self.RB_Show.clear()
        self.RB_Show.setText(msg)

    def template1_event(self):
        self.save = self.save + 1
        self.save = self.save % 2
        idx = 1
        if self.save == 1:
            self.graphic.draw_rect = True
            self.ROI.setEnabled(False)
        else:
            self.graphic.draw_rect = False
            self.graphic.rect.hide()
            if self.graphic.pixmap_rect is not None:
                img = self.graphic.org
                x1 = int(self.graphic.pixmap_rect.x())
                x2 = int(self.graphic.pixmap_rect.x() + self.graphic.pixmap_rect.width())
                y1 = int(self.graphic.pixmap_rect.y())
                y2 = int(self.graphic.pixmap_rect.y() + self.graphic.pixmap_rect.height())
                img = img[y1:y2, x1:x2, :]
                filename = './../sources/KR_C/temp/new/temp.bmp'
                cv2.imwrite(filename, img)
            self.ROI.setEnabled(True)
if __name__ == "__main__":
    #固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    #初始化
    myWin = TemperatureMeasureInitialServer()
    #将窗口控件显示在屏幕上
    myWin.show()
    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
