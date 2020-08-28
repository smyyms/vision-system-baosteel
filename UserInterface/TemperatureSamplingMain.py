# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from UserInterface.ProbeDetectMain import ProbeDetectServer
from UserInterface.TemperatureMeasureMain import TemperatureMeasureServer
from UserInterface.TemperatureSampling import Ui_TemperatureSampling


class TemperatureSamplingServer(QMainWindow, Ui_TemperatureSampling):
    def __init__(self, parent=None):
        super(TemperatureSamplingServer, self).__init__(parent)
        self.setupUi(self)
        self.UI_ProbeDetect = ProbeDetectServer()
        self.UI_TemperatureMeasure = TemperatureMeasureServer()

        self.Probe.clicked.connect(self.ProbeDetect_show)
        self.TemperatureMeasure.clicked.connect(self.TemperatureMeasure_show)

    def ProbeDetect_show(self):
        self.UI_ProbeDetect.show()

    def TemperatureMeasure_show(self):
        self.UI_TemperatureMeasure.show()


if __name__ == "__main__":
    #固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    #初始化
    myWin = TemperatureSamplingServer()
    #将窗口控件显示在屏幕上
    myWin.show()
    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
