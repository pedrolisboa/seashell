import numpy as np
from PySide2 import QtCore
from queue import Queue
import importlib
import sys

class Model(QtCore.QObject):
    loaded = QtCore.Signal()
    def __init__(self, config, model_name, model_plot=None):
        super(Model, self).__init__()

        self.decision_size = config["decisionSize"]
        self.model = self.load_model(config)
        self.q = Queue()
        self.name = model_name

        self.loaded.connect(self._update)
        self.table = None
        self.model_plot = model_plot

    def load_model(self, config):
        class_path = config["model_class"]
        class_path = class_path.split(".")
        
        model_class = class_path[-1]
        if len(class_path) > 1:
            class_module_name = class_path[:-1].join(".")
            module = importlib.import_module(class_module_name)
        else:
            module = sys.modules[__name__]

        model = getattr(module, model_class)(config)
        return model

    def update(self):
        if self.q.qsize() >= self.decision_size:
            self.loaded.emit()


    def _update(self):
        specs = list()
        for i in range(self.decision_size):
            specs.append(self.q.get())
        specs = np.vstack(specs)
        value = self.forward(specs)

        if self.table is not None:
            self.table.updateValue(self, value)

        if self.model_plot is not None:
            self.model_plot.data_sink.emit(self.name, value)


    def forward(self, specs):
        # TODO update
        # specs [n x n_fft]
        return self.model.predict(specs)
        # placeholder_index = self.decision_size//2
        # placeholder_freq = self.placeholder_freq
        # return specs[placeholder_index, placeholder_freq]

class BaseModel():
    def __init__(self, config):
        self.decision_size = config["decisionSize"]
        self.model_path = config["model_path"]
        
        self.placeholder_freq = np.random.randint(0, 100)

        ##############################################
        # TODO logica de construção do modelo
        # self.model = load...
        #############################################

    def predict(self, specs):
        placeholder_index = self.decision_size//2
        placeholder_freq = self.placeholder_freq
        return specs[placeholder_index, placeholder_freq]