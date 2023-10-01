# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'version2.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)
        MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button_file = QtWidgets.QPushButton(self.centralwidget)
        self.button_file.setGeometry(QtCore.QRect(60, 60, 120, 80))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.button_file.setFont(font)
        self.button_file.setObjectName("button_file")
        self.button_analyze = QtWidgets.QPushButton(self.centralwidget)
        self.button_analyze.setGeometry(QtCore.QRect(330, 60, 120, 80))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.button_analyze.setFont(font)
        self.button_analyze.setObjectName("button_analyze")
        self.radioButton_yes = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_yes.setGeometry(QtCore.QRect(520, 60, 120, 80))
        self.radioButton_yes.setObjectName("radioButton_yes")
        self.radioButton_no = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_no.setGeometry(QtCore.QRect(670, 60, 120, 80))
        self.radioButton_no.setObjectName("radioButton_no")
        self.label_frame = QtWidgets.QLabel(self.centralwidget)
        self.label_frame.setGeometry(QtCore.QRect(60, 170, 400, 225))
        self.label_frame.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.label_frame.setObjectName("label_frame")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(60, 470, 871, 251))
        self.tableWidget.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.tableWidget.setFrameShadow(QtWidgets.QFrame.Raised)
        self.tableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tableWidget.setTextElideMode(QtCore.Qt.ElideRight)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setRowCount(4)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 1, item)
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(520, 170, 401, 131))
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        self.button_web = QtWidgets.QPushButton(self.centralwidget)
        self.button_web.setGeometry(QtCore.QRect(810, 70, 111, 61))
        self.button_web.setObjectName("button_web")
        self.label_start = QtWidgets.QLabel(self.centralwidget)
        self.label_start.setGeometry(QtCore.QRect(520, 320, 101, 31))
        self.label_start.setAlignment(QtCore.Qt.AlignCenter)
        self.label_start.setObjectName("label_start")
        self.label_end = QtWidgets.QLabel(self.centralwidget)
        self.label_end.setGeometry(QtCore.QRect(520, 370, 101, 31))
        self.label_end.setAlignment(QtCore.Qt.AlignCenter)
        self.label_end.setObjectName("label_end")
        self.button_video = QtWidgets.QPushButton(self.centralwidget)
        self.button_video.setGeometry(QtCore.QRect(650, 320, 111, 81))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.button_video.setFont(font)
        self.button_video.setObjectName("button_video")
        self.label_framecnt = QtWidgets.QLabel(self.centralwidget)
        self.label_framecnt.setGeometry(QtCore.QRect(336, 430, 111, 20))
        self.label_framecnt.setAlignment(QtCore.Qt.AlignCenter)
        self.label_framecnt.setObjectName("label_framecnt")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_file.setText(_translate("MainWindow", "file"))
        self.button_analyze.setText(_translate("MainWindow", "analyze"))
        self.radioButton_yes.setText(_translate("MainWindow", "Report_yes"))
        self.radioButton_no.setText(_translate("MainWindow", "Report_not"))
        self.label_frame.setText(_translate("MainWindow", "TextLabel"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "1"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "2"))
        item = self.tableWidget.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "3"))
        item = self.tableWidget.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "5"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "車牌號碼"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "是否檢測"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "檢測結果"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "HC(ppm)"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "CO(%)"))
        item = self.tableWidget.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "CC(排氣量)"))
        item = self.tableWidget.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "行程別"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.item(0, 0)
        item.setText(_translate("MainWindow", "123"))
        item = self.tableWidget.item(0, 1)
        item.setText(_translate("MainWindow", "345"))
        self.tableWidget.setSortingEnabled(__sortingEnabled)
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("MainWindow", "car_1"))
        item = self.listWidget.item(1)
        item.setText(_translate("MainWindow", "car_2"))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.button_web.setText(_translate("MainWindow", "Web"))
        self.label_start.setText(_translate("MainWindow", "start"))
        self.label_end.setText(_translate("MainWindow", "end"))
        self.button_video.setText(_translate("MainWindow", "cut"))
        self.label_framecnt.setText(_translate("MainWindow", "second (s)"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())