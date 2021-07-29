

# Form implementation generated from reading ui file 'seashell2.ui',
# licensing of 'seashell2.ui' applies.
#
# Created: Tue Mar  9 02:45:24 2021
#      by: pyside2-uic  running on PySide2 5.13.2
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets
from plots import SpectrogramWidget, TimePlot, ModelPlot
from utils import FolderTree, ModelTree, ModelTable, ModelInfo, AudioSlider

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(928, 728)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 190, 361, 21))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(350, 0, 20, 681))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.playButton = QtWidgets.QPushButton(self.centralwidget)
        self.playButton.setGeometry(QtCore.QRect(80, 180, 51, 21))
        self.playButton.setObjectName("playButton")
        self.pauseButton = QtWidgets.QPushButton(self.centralwidget)
        self.pauseButton.setGeometry(QtCore.QRect(150, 180, 51, 21))
        self.pauseButton.setObjectName("pauseButton")
        self.resetbutton = QtWidgets.QPushButton(self.centralwidget)
        self.resetbutton.setGeometry(QtCore.QRect(220, 180, 51, 21))
        self.resetbutton.setObjectName("resetbutton")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(360, 0, 561, 681))
        self.frame.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.specPlot = SpectrogramWidget(self.frame)
        self.specPlot.setGeometry(QtCore.QRect(10, 10, 541, 671))
        self.specPlot.setObjectName("specPlot")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(0, 510, 361, 171))
        self.frame_2.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.timePlot = TimePlot(self.frame_2)
        self.timePlot.setGeometry(QtCore.QRect(10, 10, 331, 151))
        self.timePlot.setObjectName("timePlot")
        self.widget_3 = TimePlot(self.timePlot)
        self.widget_3.setGeometry(QtCore.QRect(340, 300, 611, 531))
        self.widget_3.setObjectName("widget_3")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 361, 181))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.audioTree = FolderTree(self.tab)
        self.audioTree.setGeometry(QtCore.QRect(0, 0, 351, 111))
        self.audioTree.setObjectName("audioTree")
        self.audioTree.header().setVisible(False)
        self.loadFileLabel = QtWidgets.QLabel(self.tab)
        self.loadFileLabel.setGeometry(QtCore.QRect(0, 131, 261, 20))
        self.loadFileLabel.setText("")
        self.loadFileLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.loadFileLabel.setMargin(3)
        self.loadFileLabel.setIndent(-5)
        self.loadFileLabel.setObjectName("loadFileLabel")
        self.fileLoadButton = QtWidgets.QPushButton(self.tab)
        self.fileLoadButton.setGeometry(QtCore.QRect(270, 130, 80, 21))
        self.fileLoadButton.setObjectName("fileLoadButton")
        self.fileTimer = AudioSlider(self.tab)
        self.fileTimer.setGeometry(QtCore.QRect(10, 110, 271, 21))
        self.fileTimer.setOrientation(QtCore.Qt.Horizontal)
        self.fileTimer.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.fileTimer.setObjectName("fileTimer")
        self.fileTimerLabel = QtWidgets.QLabel(self.tab)
        self.fileTimerLabel.setGeometry(QtCore.QRect(280, 110, 71, 21))
        self.fileTimerLabel.setText("")
        self.fileTimerLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.fileTimerLabel.setObjectName("fileTimerLabel")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(80, 50, 81, 20))
        self.label_2.setObjectName("label_2")
        self.tpsw_check = QtWidgets.QCheckBox(self.tab_2)
        self.tpsw_check.setGeometry(QtCore.QRect(30, 120, 141, 21))
        self.tpsw_check.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.tpsw_check.setObjectName("tpsw_check")
        self.label_3 = QtWidgets.QLabel(self.tab_2)
        self.label_3.setGeometry(QtCore.QRect(90, 20, 81, 20))
        self.label_3.setObjectName("label_3")
        self.dec_combo = QtWidgets.QComboBox(self.tab_2)
        self.dec_combo.setGeometry(QtCore.QRect(180, 20, 79, 23))
        self.dec_combo.setObjectName("dec_combo")
        self.dec_combo.addItem("")
        self.dec_combo.addItem("")
        self.dec_combo.addItem("")
        self.n_fft_combo = QtWidgets.QComboBox(self.tab_2)
        self.n_fft_combo.setGeometry(QtCore.QRect(180, 50, 79, 23))
        self.n_fft_combo.setObjectName("n_fft_combo")
        self.n_fft_combo.addItem("")
        self.n_fft_combo.addItem("")
        self.cutoffBox = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.cutoffBox.setGeometry(QtCore.QRect(180, 90, 66, 24))
        self.cutoffBox.setMinimum(-9999.0)
        self.cutoffBox.setMaximum(9999.99)
        self.cutoffBox.setObjectName("cutoffBox")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(260, 90, 31, 20))
        self.label_5.setObjectName("label_5")
        self.cutoffCheck = QtWidgets.QCheckBox(self.tab_2)
        self.cutoffCheck.setGeometry(QtCore.QRect(50, 90, 121, 21))
        self.cutoffCheck.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.cutoffCheck.setObjectName("cutoffCheck")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.label_9 = QtWidgets.QLabel(self.tab_3)
        self.label_9.setGeometry(QtCore.QRect(210, 20, 81, 20))
        self.label_9.setObjectName("label_9")
        self.label_8 = QtWidgets.QLabel(self.tab_3)
        self.label_8.setGeometry(QtCore.QRect(80, 20, 81, 20))
        self.label_8.setObjectName("label_8")
        self.refresh_combo = QtWidgets.QComboBox(self.tab_3)
        self.refresh_combo.setGeometry(QtCore.QRect(160, 20, 41, 23))
        self.refresh_combo.setObjectName("refresh_combo")
        self.refresh_combo.addItem("")
        self.refresh_combo.addItem("")
        self.refresh_combo.addItem("")
        self.refresh_combo.addItem("")
        self.refresh_combo.addItem("")
        self.tabWidget.addTab(self.tab_3, "")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(0, 500, 361, 21))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.tabWidget_2 = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget_2.setGeometry(QtCore.QRect(0, 200, 361, 311))
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.modelTable = ModelTable(self.tab_4)
        self.modelTable.setGeometry(QtCore.QRect(0, 0, 181, 131))
        self.modelTable.setAlternatingRowColors(True)
        self.modelTable.setGridStyle(QtCore.Qt.DashLine)
        self.modelTable.setRowCount(0)
        self.modelTable.setColumnCount(2)
        self.modelTable.setObjectName("modelTable")
        self.modelTable.setColumnCount(2)
        self.modelTable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.modelTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.modelTable.setHorizontalHeaderItem(1, item)
        self.modelTable.horizontalHeader().setVisible(True)
        self.modelTree = ModelTree(self.tab_4)
        self.modelTree.setGeometry(QtCore.QRect(190, 0, 161, 131))
        self.modelTree.setHeaderHidden(False)
        self.modelTree.setObjectName("modelTree")
        self.modelTree.header().setVisible(True)
        self.modelTree.header().setCascadingSectionResizes(False)
        self.modelInfo = ModelInfo(self.tab_4)
        self.modelInfo.setGeometry(QtCore.QRect(0, 140, 351, 131))
        self.modelInfo.setObjectName("modelInfo")
        self.tabWidget_2.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.frame_3 = QtWidgets.QFrame(self.tab_5)
        self.frame_3.setGeometry(QtCore.QRect(0, 0, 461, 361))
        self.frame_3.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.modelPlot = ModelPlot(self.frame_3)
        self.modelPlot.setGeometry(QtCore.QRect(0, 10, 341, 261))
        self.modelPlot.setObjectName("modelPlot")
        self.tabWidget_2.addTab(self.tab_5, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 928, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "MainWindow", None, -1))
        self.playButton.setText(QtWidgets.QApplication.translate("MainWindow", "Play", None, -1))
        self.pauseButton.setText(QtWidgets.QApplication.translate("MainWindow", "Pause", None, -1))
        self.resetbutton.setText(QtWidgets.QApplication.translate("MainWindow", "Reset", None, -1))
        self.fileLoadButton.setText(QtWidgets.QApplication.translate("MainWindow", "Unload", None, -1))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QtWidgets.QApplication.translate("MainWindow", "Files", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("MainWindow", "Window Size", None, -1))
        self.tpsw_check.setText(QtWidgets.QApplication.translate("MainWindow", "TPSW", None, -1))
        self.label_3.setText(QtWidgets.QApplication.translate("MainWindow", "Decimation", None, -1))
        self.dec_combo.setItemText(0, QtWidgets.QApplication.translate("MainWindow", "1", None, -1))
        self.dec_combo.setItemText(1, QtWidgets.QApplication.translate("MainWindow", "3", None, -1))
        self.dec_combo.setItemText(2, QtWidgets.QApplication.translate("MainWindow", "4", None, -1))
        self.n_fft_combo.setItemText(0, QtWidgets.QApplication.translate("MainWindow", "1024", None, -1))
        self.n_fft_combo.setItemText(1, QtWidgets.QApplication.translate("MainWindow", "2048", None, -1))
        self.label_5.setText(QtWidgets.QApplication.translate("MainWindow", "(dB)", None, -1))
        self.cutoffCheck.setText(QtWidgets.QApplication.translate("MainWindow", "Spectrum Cutoff", None, -1))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QtWidgets.QApplication.translate("MainWindow", "LOFAR", None, -1))
        self.label_9.setText(QtWidgets.QApplication.translate("MainWindow", "ups", None, -1))
        self.label_8.setText(QtWidgets.QApplication.translate("MainWindow", "Refresh rate", None, -1))
        self.refresh_combo.setItemText(0, QtWidgets.QApplication.translate("MainWindow", "1", None, -1))
        self.refresh_combo.setItemText(1, QtWidgets.QApplication.translate("MainWindow", "5", None, -1))
        self.refresh_combo.setItemText(2, QtWidgets.QApplication.translate("MainWindow", "15", None, -1))
        self.refresh_combo.setItemText(3, QtWidgets.QApplication.translate("MainWindow", "30", None, -1))
        self.refresh_combo.setItemText(4, QtWidgets.QApplication.translate("MainWindow", "60", None, -1))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QtWidgets.QApplication.translate("MainWindow", "Configuration", None, -1))
        self.modelTable.horizontalHeaderItem(0).setText(QtWidgets.QApplication.translate("MainWindow", "Model", None, -1))
        self.modelTable.horizontalHeaderItem(1).setText(QtWidgets.QApplication.translate("MainWindow", "Output", None, -1))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), QtWidgets.QApplication.translate("MainWindow", "Model Info", None, -1))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_5), QtWidgets.QApplication.translate("MainWindow", "Model Plot", None, -1))
        self.menuFile.setTitle(QtWidgets.QApplication.translate("MainWindow", "File", None, -1))

