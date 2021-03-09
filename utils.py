import os
from PySide2.QtWidgets import QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView, QSlider
from PySide2 import QtCore

from main import MainWindow

class FolderTree(QTreeWidget):
    def __init__(self, parent=None):
        super(FolderTree, self).__init__(parent=parent)
        self.startDir = os.path.join(os.getcwd(), 'audios')
        self.fillTree()
        self.show()

    def get_path(self, entry):
        def build_path(currentDir, item):
            for f in os.listdir(currentDir):
                path = os.path.join(currentDir, f)
                if os.path.isdir(path):
                    result = build_path(path, item)
                    if result is not None:
                        return result
                else:
                    if f == entry:
                        return path
        return build_path(self.startDir, entry)

    def fillTree(self):        
        def iterate(currentDir, currentItem):            
            for f in os.listdir(currentDir):
                path = os.path.join(currentDir, f)
                if os.path.isdir(path):
                    dirItem = QTreeWidgetItem(currentItem)
                    dirItem.setText(0, f)
                    iterate(path, dirItem)
                else:
                    if f.endswith('wav'):
                        fileItem = QTreeWidgetItem(currentItem)
                        fileItem.setText(0, f)
        iterate(self.startDir, self)

class ModelTree(QTreeWidget):
    def __init__(self, parent=None):
        super(ModelTree, self).__init__(parent=parent)
        self.startDir = os.path.join(os.getcwd(), 'models', 'configs')
        self.objDir = os.path.join(os.getcwd(), 'models')
        self.setHeaderLabels(["Available Models"])

        self.fillTree()
        self.show()

    def fillTree(self):        
        def iterate(currentDir, currentItem):            
            for f in os.listdir(currentDir):
                path = os.path.join(currentDir, f)
                if os.path.isdir(path):
                    dirItem = QTreeWidgetItem(currentItem)
                    dirItem.setText(0, f)
                    iterate(path, dirItem)
                else:
                    if f.endswith('json'):
                        fileItem = QTreeWidgetItem(currentItem)
                        fileItem.setText(0, f)
        iterate(self.startDir, self)

class AudioSlider(QSlider):
    pos = QtCore.Signal(int)
    def __init__(self, parent=None, label=None):
        super(AudioSlider, self).__init__(parent=parent)

        self.pos.connect(self.updatePos)
        if label is not None:
            self.valueChanged.connect(self._print_to_label)
        self.label = label
        self.sliderPressed.connect(self._pressed)
        self.sliderReleased.connect(self._released)

        self._pressed_callback = None
        self._released_callback = None

        self.ar = None

    def _pressed(self):
        if self._pressed_callback is not None:
            self._pressed_callback(self.value())

    def _released(self):
        if self.ar is not None:
            self.ar.set_pos(self.value())
        if self._released_callback is not None:
            self._released_callback(self.value())

    def bind_callbacks(self, pressed=None, released=None):
        if pressed is not None:
            self._pressed_callback = pressed
        if released is not None:
            self._released_callback = released

    def bind_label(self, qlabel):
        if self.label is None:
            self.label = qlabel
            self.valueChanged.connect(self._print_to_label)
            return 
        self.label = qlabel

    def setVisible(self, b):
        super().setVisible(b)
        if self.label is not None:
            self.label.setVisible(b)

    def bind_audio_receiver(self, ar):
        ar.pos_sink = self.pos
        self.frames = ar.frames
        self.rate = ar.or_rate
        self.width = ar.width

        self.ar = ar
        self.setMaximum(self.frames)

    def _print_to_label(self, value):
        if self.rate is not None:
            seconds = value/self.rate
            minutes = seconds // 60
            rem_seconds = seconds % 60

            self.label.setText("%02d:%02d" % (minutes, round(rem_seconds)))

    def updatePos(self, value):
        self.setSliderPosition(value)
        if value >= (self.frames//self.ar.or_rate)*self.ar.or_rate:
            self.ar.end_sink.emit()

class ModelInfo(QTreeWidget):
    def __init__(self, parent=None):
        super(ModelInfo, self).__init__(parent=parent)
        self.setHeaderLabels(["Model Properties", "Value"])
        header = self.header()       
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.show()

    def fillTree(self, name, config):    
        self.clear()

        def iterate(config, currentItem):            
            for key in config.keys():
                keyItem = QTreeWidgetItem(currentItem)
                keyItem.setText(0, key)

                item = config[key]
                if isinstance(item, dict):
                    iterate(item, keyItem)
                else:
                    keyItem.setText(1, str(item))
        nameItem = QTreeWidgetItem(self)
        nameItem.setText(0, name)                
        iterate(config, nameItem)
        self.expandToDepth(2)



class ModelTable(QTableWidget):
    def __init__(self, parent, *args):
        super(ModelTable, self).__init__(parent, *args)

        self.mylist = []
        self.header = ["Model", "Output"]
        self.setEditTriggers(QTableWidget.NoEditTriggers)

        header = self.horizontalHeader()       
        header.setSectionResizeMode(QHeaderView.Stretch)



    def cell(self,var=""):
            item = QTableWidgetItem()
            item.setText(var)
            return item

    def addModel(self, model):
        model_name = model.name
        n_models = self.rowCount()
        self.insertRow(n_models)
        self.setItem(n_models, 0, QTableWidgetItem(model_name))
        self.setItem(n_models, 1, QTableWidgetItem(""))

        model.table = self

    def exists_entry(self, model_name):
        i_model = self.find_model_index(model_name)
        print(i_model)
        return i_model >= 0
        

    def removeModel(self, model_name):
        i_model = self.find_model_index(model_name)
        if i_model < 0:
            return
        
        self.removeRow(i_model)


    def updateValue(self, model, value):
        model_name = model.name

        i_model = self.find_model_index(model_name)
                    
        if i_model < 0:
            return
        # n_models = self.rowCount()
        # for i_model in range(n_models):
        #     row_name = self.item(i_model, 0).text()

        #     if row_name == model_name:
        self.setItem(i_model, 1, QTableWidgetItem("%.2f" % value))
                # return 

    def find_model_index(self, model_name):
        n_models = self.rowCount()
        for i_model in range(n_models):
            row_name = self.item(i_model, 0).text()

            if row_name == model_name:
                return i_model
        return -1

            
            