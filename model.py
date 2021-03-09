import numpy as np
from PySide2 import QtCore
from queue import Queue

class Model(QtCore.QObject):
    loaded = QtCore.Signal()
    def __init__(self, decision_size=1, name='', model_path='', model_plot=None):
        super(Model, self).__init__()

        self.decision_size = decision_size
        self.model_path = model_path
        self.q = Queue()
        self.name = name

        self.loaded.connect(self._update)

        self.table = None

        self.model_plot = model_plot

        self.placeholder_freq = np.random.randint(0, 100)


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
        placeholder_index = self.decision_size//2
        placeholder_freq = self.placeholder_freq
        return specs[placeholder_index, placeholder_freq]

