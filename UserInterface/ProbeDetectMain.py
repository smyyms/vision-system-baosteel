# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox,QInputDialog,qApp,QDialog
from PyQt5.QtGui import QImage, QPixmap,QPen,QBrush,QPainter
from PyQt5.QtCore import Qt,QRect,pyqtSignal
from UserInterface.ProbeDetect import Ui_ProbeDetect
from UserInterface.ProbeDetectInitialMain import ProbeDetectInittalServer


#Image Process
from bgimage.KR_A_Utils import *

import socket
import struct
import ctypes
import datetime
import threading
import time
import sys,os


VERBOSE=True

# TCP/IP
HOST = socket.gethostname()
# PORT = 12356
# HOST = '192.168.0.1'
PORT = 2000
ADDR = (HOST, PORT)
BUFSIZE = 1024
FLAG_CONNECT = True

StatusShowList = []
ServerRecvList = []
ServerSentList = []


class ProbeDetectServer(QMainWindow, Ui_ProbeDetect):
    def __init__(self, parent=None):
        super(ProbeDetectServer, self).__init__(parent)
        self.setupUi(self)
        self.init_Ui = ProbeDetectInittalServer()
        #self.showMaximized()

        self.sub_list = []
        # 显示处理后的图像,这里也得用线程处理
        # camera_ip_address = self.init_Ui.IP_address
        # image_ori = read_image_from_camera(camera_ip_address)
        image_ori = cv2.imread(KR_A_source_file + "/" + KR_A_demo_filename)
        cvRGBImg = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(cvRGBImg.data, cvRGBImg.shape[1], cvRGBImg.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg).scaled(self.ImageShow.width(),self.ImageShow.height())
        #image = QPixmap(KR_A_source_file + "/" + KR_A_demo_filename).scaled(self.ImageShow.width(),self.ImageShow.height())
        self.ImageShow.setPixmap(pixmap)

        #连接部分
        self.listen_sock = None
        self.accept_thread = None
        self.s_th = None
        self.client_socket_list = list()

        self.StartServer.clicked.connect(self.socket_open)
        self.CloseServer.clicked.connect(self.socket_close)
        self.pushButton.clicked.connect(self.init_ui_show)
        #子线程消息显示,信号槽机制
        self.signal_show_recv_msg.connect(self.show_recv_msg)
        self.signal_show_sent_msg.connect(self.show_sent_msg)
        self.signal_show_status_msg.connect(self.show_status_msg)
        self.signal_status_msg_clear.connect(self.clear_status_msg)
        self.signal_show_image.connect(self.show_image)


    def init_ui_show(self):
        self.init_Ui.show()

    def socket_open(self):

        """
        TCP服务端开启方法
        :return: 
        """"""
        """
        self.StartServer.setEnabled(False)

        startTime = None
        endTime = None

        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_sock.settimeout(5.0)  # 设定超时时间后，socket其实内部变成了非阻塞，但有一个超时时间
        try:
            self.listen_sock.bind(ADDR)
            self.listen_sock.listen(2)
            # time.sleep(0.2)
        except Exception as ret:
            msg = "连接出现错误,正在关闭！"
            self.signal_show_status_msg.emit(msg)
            self.socket_close()
        else:
            msg = "准备建立新的TCP/IP连接..."
            self.signal_show_status_msg.emit(msg)
            self.accept_thread = threading.Thread(target=self.accept_concurrency)
            self.accept_thread.setDaemon(True)
            self.accept_thread.start()

    def accept_concurrency(self):
        """
        创建监听线程，使GUI主线程继续运行，以免造成未响应
        """

        while True:
            try:
                connected_sock, client_addr = self.listen_sock.accept()
            except socket.timeout:

                length = len(self.sub_list)
                # while length:
                #     sub = sub_list.pop(0)
                #     connected_sock, client_addr = self.client_socket_list.pop(0)
                #     sub_id = sub.ident  # 进程ID
                #     sub.join(0.1)  # 等待线程结束，0.1秒
                #     if sub.isAlive():
                #         sub_list.append(sub)
                #         self.client_socket_list.append((connected_sock, client_addr))
                #     else:
                #         msg1 = '结束子线程：' + str(sub_id)
                #         self.signal_show_status_msg.emit(msg1)
                #         msg2 = '等待新的TCP/IP连接...'
                #         self.signal_show_status_msg.emit(msg2)
                #     length -= 1
            else:
                self.client_socket_list.append((connected_sock, client_addr))
                msg = "连接已建立."
                self.signal_show_status_msg.emit(msg)
                msg = "客户端地址为：" + str(client_addr)
                self.signal_show_status_msg.emit(msg)
                self.s_th = threading.Thread(target=self.handle, name='sub thread', args=(connected_sock,))
                # 它继承了listen_socket的阻塞/非阻塞特性，因为listen_socket是非阻塞的，所以它也是非阻塞的
                # 要让他变为阻塞，所以要调用setblocking
                connected_sock.setblocking(1)
                self.s_th.setDaemon(True)
                self.s_th.start()
                self.sub_list.append(self.s_th)
                # print(len(sub_list),len(self.client_socket_list))


    def socket_close(self):
        #self.CloseServer.setEnabled(False)
        self.ServerRecvShow.clear()
        self.ServerSentShow.clear()
        self.StatusShow.clear()
        if len(self.client_socket_list):
            try:
                for client,address in self.client_socket_list:
                    client.close()
                    self.client_socket_list.remove((client, address))
                self.listen_sock.close()
                self.StartServer.setEnabled(True)
            except Exception as ret:
                pass
        else:
            msg = "无客户端连接！"
            self.signal_show_status_msg.emit(msg)
            self.StartServer.setEnabled(True)
        try:
            #关闭线程
            for sub in self.sub_list:
                sub.join(0.1)
            #stopThreading.stop_thread(self.s_th)
            #stopThreading.stop_thread(self.accept_thread)
        except Exception:
            pass

    def handle(self, connected_sock):
        """
        功能函数，为每个tcp连接创建一个线程；
        使用子线程用于创建连接，使每个tcp client可以单独地与server通信
        """

        while True:
            data = connected_sock.recv(BUFSIZE)
            recv_data_pack = RecvStruct()
            send_data_pack = SendStruct()
            if len(data) > 0:
                cur_thread = threading.current_thread()
                msg1 = "接收到客户端指令，当前线程ID为：" + str(cur_thread.ident) + "，正在解析...'"
                self.signal_show_status_msg.emit(msg1)
                time.sleep(1)
                recv_data_buff = struct.unpack('BBBB', data)
                # print(len(recv_data_buff),'    ',recv_data_buff)
                string, flag = check(recv_data_buff)
                if not flag:
                    self.signal_show_status_msg.emit(string)
                    msg = "接收客户端指令格式有误！重新等待指令..."
                    self.signal_show_status_msg.emit(msg)
                    continue
                recv_data_pack.heartbeat = int(str(recv_data_buff[0]), 16)
                recv_data_pack.camera_id = int(str(recv_data_buff[1]), 16)
                recv_data_pack.status = int(str(recv_data_buff[2]), 16)
                recv_data_pack.reserve = int(str(recv_data_buff[3]), 16)
                msg = "接收到客户端指令：" + str(recv_data_pack.heartbeat) + str(recv_data_pack.camera_id) + str(recv_data_pack.status) + str(recv_data_pack.reserve)
                self.signal_show_recv_msg.emit(msg)
                if recv_data_pack.heartbeat == 0:
                    msg = "接收断开心跳指令，正在断开客户端连接..."
                    self.signal_show_status_msg.emit(msg)
                    connected_sock.close()
                    break
                else:
                    send_data_pack.heartbeat = recv_data_pack.heartbeat

                if recv_data_pack.status == 0:
                    msg = '控制指令：待命'
                    self.signal_show_status_msg.emit(msg)
                    send_data_pack.status = 0
                elif recv_data_pack.status == 2:
                    msg = '控制指令：复位'
                    self.signal_show_status_msg.emit(msg)
                    send_data_pack.status = 2
                elif recv_data_pack.status == 1:
                    msg = '控制指令：作业'
                    self.signal_show_status_msg.emit(msg)
                    send_data_pack.status = 1
                    msg = '正在处理.......'
                    self.signal_show_status_msg.emit(msg)
                    #camera_ip_address = self.init_Ui.IP_address
                    #image_ori = read_image_from_camera(camera_ip_address)
                    image_ori = read_image_from_file(KR_A_source_file + "/" + KR_A_demo_filename)
                    result = KR_A_process(image_ori)

                    image = QPixmap("./../sources/KR_A_display.png").scaled(self.ImageShow.width(), self.ImageShow.height())
                    self.signal_show_image.emit(image)

                    target_id = -1
                    for i in range(PROBE_NUM):
                        order_id = ORDER[i]
                        if result[order_id][1] == 1:
                            target_id = i + 1
                            break
                    target_res_num = 0
                    arrange = 0
                    for i in range(PROBE_NUM):
                        order_id = ORDER[i]
                        send_data_pack.data[i][1] = result[order_id][1]
                        send_data_pack.data[i][2], send_data_pack.data[i][3] = result[order_id][2], result[order_id][3]
                        if result[order_id][1] != 0 and result[order_id][1] != -1 and result[order_id][2] != -1:
                            target_res_num += 1
                            arrange |= 2 ** i
                            # print(arrange)
                    send_data_pack.target[0], send_data_pack.target[1] = target_id, target_res_num
                    send_data_pack.target[2] = arrange
                # status = 0
                # TODO status
                s1 = struct.Struct('B')
                s2 = struct.Struct('>bbhh')
                s3 = struct.Struct('>bbI')
                send_buff = ctypes.create_string_buffer(4 * s1.size + 15 * s2.size + s3.size)
                packed_data_s1 = s1.pack_into(send_buff, 0, send_data_pack.heartbeat)
                packed_data_s2 = s1.pack_into(send_buff, s1.size, send_data_pack.camera_id)
                packed_data_s3 = s1.pack_into(send_buff, 2 * s1.size, send_data_pack.program_id)
                packed_data_s4 = s1.pack_into(send_buff, 3 * s1.size, send_data_pack.status)
                # packed_data_s5 = s1.pack_into(send_buff, 4 * s1.size, send_data_pack.reserve)
                for i in range(PROBE_NUM):
                    s2.pack_into(send_buff, 4 * s1.size + i * s2.size, *send_data_pack.data[i])
                s3.pack_into(send_buff, 4 * s1.size + 15 * s2.size, *send_data_pack.target)

                msg = str(send_data_pack.heartbeat) + " " + str(send_data_pack.camera_id) + " " + \
                      str( send_data_pack.program_id) + " " + str(send_data_pack.status) + " " + \
                    str(send_data_pack.data) + " " + str( send_data_pack.target)
                self.signal_show_sent_msg.emit(msg)
                connected_sock.sendall(send_buff)
            else:
                msg1 = '接收客户端指令失败，关闭SOCKET连接!'
                self.signal_show_status_msg.emit(msg1)
                connected_sock.close()
                msg2 = '等待新的TCP/IP连接...'
                self.signal_show_status_msg.emit(msg2)
                break

    def show_recv_msg(self, msg):
        """
        功能函数，向接收区写入数据的方法
        信号-槽触发
        tip：PyQt程序的子线程中，直接向主线程的界面传输字符是不符合安全原则的
        """
        self.ServerRecvShow.clear()
        self.ServerRecvShow.setText(msg)

    def show_sent_msg(self, msg):
        # 显示发送信息
        self.ServerSentShow.clear()
        self.ServerSentShow.setText(msg)

    def show_status_msg(self, msg):
        # 显示状态信息
        self.StatusShow.append(msg)

    def clear_status_msg(self, msg):
        self.StatusShow.clear()
        self.ServerSentShow.clear()
        self.ServerRecvShow.clear()
        self.StatusShow.setText(msg)

    def show_image(self, image):
        self.ImageShow.clear()
        self.ImageShow.setPixmap(image)



if __name__ == "__main__":
    #固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    #初始化
    myWin = ProbeDetectServer()
    #将窗口控件显示在屏幕上
    myWin.show()
    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
