from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
import webbrowser
from selenium.webdriver.common.by import By
from video_controller import video_controller
import time
from moviepy.editor import *
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from BUG import Bug

from UI import Ui_MainWindow
#https://mobile.epa.gov.tw/Motor/query/Query_Check.aspx
class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        row_position = self.ui.tableWidget.rowCount()  # Get the current row count
        self.ui.tableWidget.insertRow(row_position)   # Insert a new row

        # Now, populate the cells in the new row with items
        for i in range(self.ui.tableWidget.rowCount()):
            for j in range(self.ui.tableWidget.columnCount()):
                item = QtWidgets.QTableWidgetItem()
                self.ui.tableWidget.setItem(i, j, item)

    def setup_control(self):
        self.ui.button_web.clicked.connect(self.open_web)
        self.ui.button_video.clicked.connect(self.video_cut)
        self.ui.button_file.clicked.connect(self.open_file)

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi)") # start path        
        self.video_path = filename
        self.video_controller = video_controller(video_path=self.video_path,
                                                 ui=self.ui)
        # self.ui.label_filepath.setText(f"video path: {self.video_path}")
        

    def video_cut(self):
        video = VideoFileClip("1.mp4")
        # output = video.subclip(self.ui.label_start,self.ui.label_end)  
        output = video.subclip(7, 15)
        output.write_videofile("output_2.mp4",temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
        pass




    def open_web(self):
        self.web = Bug()
        self.web.start_B()

        _translate = QtCore.QCoreApplication.translate
        for i in range(6):
            item = self.ui.tableWidget.item(0, i)
            item.setText(_translate("MainWindow", self.web.search_3[i]))

        self.web.open_web()
        time.sleep(2)
        # for i in range(1,3):
        #     self.ui.tableWidget.item(1, i).setText(f"車牌號碼:{self.web.vehicle}")
        # self.ui.tableWidget.item(0, 1).setText('hi1')
        # self.ui.tableWidget.item(0, 0).setText('hi2')
        # self.ui.tableWidget.item(0, 2).setText('hi3')
        # self.ui.tableWidget.item(0, 2).setText('hi3')
        # self.ui.tableWidget.item(0, 0).setText('hiiiiiiiii')
        
        # _translate = QtCore.QCoreApplication.translate
        # for i in range(6):
        #     item = self.ui.tableWidget.item(0, i)
        #     item.setText(_translate("MainWindow", self.web.search_3[i]))
        print("HI")
