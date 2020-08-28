# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ProbDetectInitial.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class Ui_ProbeDetectInitial(object):

    signal_show_point1 = pyqtSignal(str)
    signal_show_point2 = pyqtSignal(str)
    def setupUi(self, ProbeDetectInitial):
        ProbeDetectInitial.setObjectName("ProbeDetectInitial")
        ProbeDetectInitial.resize(944, 671)
        self.centralwidget = QtWidgets.QWidget(ProbeDetectInitial)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(540, 80, 401, 521))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.Status = QtWidgets.QLabel(self.groupBox)
        self.Status.setGeometry(QtCore.QRect(10, 10, 101, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.Status.setFont(font)
        self.Status.setObjectName("Status")
        self.IP = QtWidgets.QTextEdit(self.groupBox)
        self.IP.setGeometry(QtCore.QRect(13, 63, 371, 61))
        self.IP.setObjectName("IP")
        self.Point1 = QtWidgets.QPushButton(self.groupBox)
        self.Point1.setGeometry(QtCore.QRect(30, 140, 111, 41))
        self.Point1.setObjectName("Point1")
        self.Point2 = QtWidgets.QPushButton(self.groupBox)
        self.Point2.setGeometry(QtCore.QRect(240, 140, 111, 41))
        self.Point2.setObjectName("Point2")
        self.Point1Show = QtWidgets.QTextBrowser(self.groupBox)
        self.Point1Show.setGeometry(QtCore.QRect(20, 200, 151, 81))
        self.Point1Show.setObjectName("Point1Show")
        self.Point2Show = QtWidgets.QTextBrowser(self.groupBox)
        self.Point2Show.setGeometry(QtCore.QRect(220, 200, 151, 81))
        self.Point2Show.setObjectName("Point2Show")
        self.layoutWidget = QtWidgets.QWidget(self.groupBox)
        self.layoutWidget.setGeometry(QtCore.QRect(2, 270, 403, 271))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.template1 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template1.setFont(font)
        self.template1.setObjectName("template1")
        self.horizontalLayout.addWidget(self.template1)
        self.template2 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template2.setFont(font)
        self.template2.setObjectName("template2")
        self.horizontalLayout.addWidget(self.template2)
        self.template3 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template3.setFont(font)
        self.template3.setObjectName("template3")
        self.horizontalLayout.addWidget(self.template3)
        self.template4 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template4.setFont(font)
        self.template4.setObjectName("template4")
        self.horizontalLayout.addWidget(self.template4)
        self.template5 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template5.setFont(font)
        self.template5.setObjectName("template5")
        self.horizontalLayout.addWidget(self.template5)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.template6 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template6.setFont(font)
        self.template6.setObjectName("template6")
        self.horizontalLayout_2.addWidget(self.template6)
        self.template7 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template7.setFont(font)
        self.template7.setObjectName("template7")
        self.horizontalLayout_2.addWidget(self.template7)
        self.template8 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template8.setFont(font)
        self.template8.setObjectName("template8")
        self.horizontalLayout_2.addWidget(self.template8)
        self.template9 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template9.setFont(font)
        self.template9.setObjectName("template9")
        self.horizontalLayout_2.addWidget(self.template9)
        self.template10 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template10.setFont(font)
        self.template10.setObjectName("template10")
        self.horizontalLayout_2.addWidget(self.template10)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.template11 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template11.setFont(font)
        self.template11.setObjectName("template11")
        self.horizontalLayout_3.addWidget(self.template11)
        self.template12 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template12.setFont(font)
        self.template12.setObjectName("template12")
        self.horizontalLayout_3.addWidget(self.template12)
        self.template13 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template13.setFont(font)
        self.template13.setObjectName("template13")
        self.horizontalLayout_3.addWidget(self.template13)
        self.template14 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template14.setFont(font)
        self.template14.setObjectName("template14")
        self.horizontalLayout_3.addWidget(self.template14)
        self.template15 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(10)
        self.template15.setFont(font)
        self.template15.setObjectName("template15")
        self.horizontalLayout_3.addWidget(self.template15)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.titleName = QtWidgets.QLabel(self.centralwidget)
        self.titleName.setGeometry(QtCore.QRect(310, 10, 390, 81))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(16)
        self.titleName.setFont(font)
        self.titleName.setObjectName("titleName")
        self.Image = QtWidgets.QLabel(self.centralwidget)
        self.Image.setGeometry(QtCore.QRect(210, 60, 131, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.Image.setFont(font)
        self.Image.setObjectName("Image")
        self.ImageShow = QtWidgets.QGraphicsView(self.centralwidget)
        self.ImageShow.setGeometry(QtCore.QRect(10, 130, 511, 471))
        self.ImageShow.setObjectName("ImageShow")
        ProbeDetectInitial.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ProbeDetectInitial)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 944, 21))
        self.menubar.setObjectName("menubar")
        ProbeDetectInitial.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ProbeDetectInitial)
        self.statusbar.setObjectName("statusbar")
        ProbeDetectInitial.setStatusBar(self.statusbar)

        self.retranslateUi(ProbeDetectInitial)
        QtCore.QMetaObject.connectSlotsByName(ProbeDetectInitial)
        self.buttonlist = [self.template1, self.template2, self.template3, self.template4, self.template5,
                           self.template6, self.template7, self.template8, self.template9, self.template10,
                           self.template11, self.template12, self.template13, self.template14, self.template15]
    def retranslateUi(self, ProbeDetectInitial):
        _translate = QtCore.QCoreApplication.translate
        ProbeDetectInitial.setWindowTitle(_translate("ProbeDetectInitial", "MainWindow"))
        self.groupBox.setTitle(_translate("ProbeDetectInitial", "初始化参数"))
        self.Status.setText(_translate("ProbeDetectInitial", "相机IP："))
        self.IP.setHtml(_translate("ProbeDetectInitial", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'楷体\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:28pt; vertical-align:sub;\">192.168.0.123</span></p></body></html>"))
        self.Point1.setText(_translate("ProbeDetectInitial", "Point1"))
        self.Point2.setText(_translate("ProbeDetectInitial", "Point2"))
        self.template1.setText(_translate("ProbeDetectInitial", "模板1"))
        self.template2.setText(_translate("ProbeDetectInitial", "模板2"))
        self.template3.setText(_translate("ProbeDetectInitial", "模板3"))
        self.template4.setText(_translate("ProbeDetectInitial", "模板4"))
        self.template5.setText(_translate("ProbeDetectInitial", "模板5"))
        self.template6.setText(_translate("ProbeDetectInitial", "模板6"))
        self.template7.setText(_translate("ProbeDetectInitial", "模板7"))
        self.template8.setText(_translate("ProbeDetectInitial", "模板8"))
        self.template9.setText(_translate("ProbeDetectInitial", "模板9"))
        self.template10.setText(_translate("ProbeDetectInitial", "模板10"))
        self.template11.setText(_translate("ProbeDetectInitial", "模板11"))
        self.template12.setText(_translate("ProbeDetectInitial", "模板12"))
        self.template13.setText(_translate("ProbeDetectInitial", "模板13"))
        self.template14.setText(_translate("ProbeDetectInitial", "模板14"))
        self.template15.setText(_translate("ProbeDetectInitial", "模板15"))
        self.titleName.setText(_translate("ProbeDetectInitial", "取样器视觉检测参数初始化"))
        self.Image.setText(_translate("ProbeDetectInitial", "相机图像"))