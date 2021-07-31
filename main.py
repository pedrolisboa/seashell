import sys
from app import Ui_MainWindow
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCore import QTimer
from PySide2 import QtCore


from processing import MicReceiver, FileReceiver
from queue import Queue


import pyqtgraph as pg

import numpy as np
import json

import sys,os
import PySide2

from model import Model

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'Qt', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

class MainWindow(QMainWindow):
    def __init__(self, config=None):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.defineConfigurationOptions(config)

        self.mr = None
        self.qs = {
            'receiver' : Queue(),
        }
        self.sig_sink = None

        self.current_audio_file = None

        # connect buttons
        self.ui.playButton.clicked.connect(self.play)
        self.ui.pauseButton.clicked.connect(self.pause)
        self.ui.resetbutton.clicked.connect(self.reset)
        self.ui.fileLoadButton.clicked.connect(self.unload_file)
        self.ui.fileLoadButton.setVisible(False)

        
        
        self.ui.fileTimer.bind_label(self.ui.fileTimerLabel)
        self.ui.fileTimer.setVisible(False)

        def release_cb(value):
            self.reset(reset_models=False, pause=True, purge_recording=False, flush_queues=True)
            self.play()
        self.ui.fileTimer.bind_callbacks(
            lambda _: self.pause(), 
            release_cb
        )


        self.ui.modelTree.itemDoubleClicked.connect(self.load_model)
        self.ui.modelTree.itemClicked.connect(self.show_model_info)
        # self.ui.modelTable.removeModel(model)

        self.ui.audioTree.itemDoubleClicked.connect(self.load_file)

        self.ui.modelTable.cellDoubleClicked.connect(self.del_model)

        self.ui.tpsw_check.toggled.connect(self.apply_tpsw)
        self.ui.cutoffBox.valueChanged.connect(self.apply_cutoff)
        self.ui.cutoffCheck.stateChanged.connect(self.apply_cutoff)

        self.lockPlotScroll(True, False)

    def load_file(self, item, col):
        audio_file = item.text(0)
        audio_file_path = self.ui.audioTree.get_path(audio_file)
        if self.current_audio_file is not None:
            self.unload_file()
        self.ui.fileLoadButton.setVisible(True)
        self.ui.fileTimer.setVisible(True)
        self.ui.loadFileLabel.setText(audio_file + " loaded")
        self.current_audio_file = audio_file_path

        self.reset(reset_models=False)

        
    def unload_file(self):
        # stop audiofile
        # clear workspace
        self.ui.fileLoadButton.setVisible(False)
        self.ui.fileTimer.setVisible(False)
        self.ui.fileTimer.setSliderPosition(0)

        self.ui.loadFileLabel.setText("")
        self.current_audio_file = None

        self.reset()

        # remove audio
        # remove tracking from slider



    def show_model_info(self, item, col):
        config_file = item.text(0)
        model_name = config_file.strip('.json')
        config = json.load(open(os.path.join(self.ui.modelTree.startDir, config_file), 'r'))

        self.ui.modelInfo.fillTree(model_name, config)
        # text = "Model: " + model_name + "\n"
        # for key in config.keys():
        #     text += "-" * 4 + "> " + key + ": " + str(config[key]) + "\n"


        # self.ui.modelInfo.setText(text)

    #temporary
    def apply_tpsw(self, checked):
        self.reset()
        self.config["tpsw"] = checked
        self.play()

    def apply_cutoff(self, value):
        if self.ui.cutoffCheck.isChecked():
            self.reset()
            self.config["spec_cutoff"] = float(value)
        else:
            print("Value" + str(value))
            self.reset()
            self.config["spec_cutoff"] = np.nan
        self.play()



    def del_model(self, row, col):
        model_name = self.ui.modelTable.item(row, col).text()

        self.ui.modelTable.removeModel(model_name)
        self.ui.specPlot.unsubscribe_model(model_name)
        self.ui.modelPlot.unsubscribe_model(model_name)

    def load_model(self, item, col):
        config_file = item.text(0)
        model_name = config_file.strip('.json')
        if self.ui.modelTable.exists_entry(model_name):
            return

        config = json.load(open(os.path.join(self.ui.modelTree.startDir, config_file), 'r'))

        model = Model(config, model_name, self.ui.modelPlot)
        
        #model limiter
        # if self.ui.modelTable.rowCount() > 0:
        #     self.del_model(0, 0)

        self.ui.modelTable.addModel(model)
        self.ui.specPlot.subscribe_model(model.name, model)
        self.ui.modelPlot.subscribe_model(model.name, model)


    def reset(self, reset_models=True, pause=True, purge_recording=True, flush_queues=True, reset_graphics=True):
        if self.sig_sink is not None:
            self.sig_sink.stop()

        
        # self.ui.modelPlot.clear_plot()
        # limit model
        # if (self.ui.modelTable.rowCount() > 0) and reset_models:
        #     self.del_model(0, 0)

        if pause:
            self.pause()

        if purge_recording:
            if self.mr is not None:
                self.mr.stop()
            if self.sig_sink is not None:
                self.sig_sink.stop()

            self.mr = None
            self.sig_sink = None
            self.rq = {
                'receiver' : Queue(),
            }
        if flush_queues:
            self.flush_queues()

        if reset_graphics:
            self.ui.specPlot.clear_plot()
            self.ui.demonPlot.clear_plot()
            #self.ui.timePlot.clear_plot()
            if not reset_models:
                self.ui.modelPlot.clear_plot(unsubscribe=False)

    def defineConfigurationOptions(self, config):
        def_fps = 30
        if config is None:
            config = {
                "rf_rate": 1000/def_fps, # ms
                "n_fft": 1024,
                "decimation": 3,
                "tpsw": True,
                "spec_cutoff": -4.00,
                "demon_maxfreq": 35*60,
                "demon_n_fft": 1024,
                "demon_overlap": 0.5 
            }
        else:
            raise NotImplementedError("User input config not implemented")

        self.ui.refresh_combo.setCurrentText(str(def_fps))
        self.ui.refresh_combo.currentTextChanged.connect(self.changeRefreshRate)

        self.ui.n_fft_combo.setCurrentText(str(config["n_fft"]))
        self.ui.dec_combo.setCurrentText(str(config["decimation"]))

        self.ui.n_fft_combo.currentTextChanged.connect(lambda x: self.updateSignalProcessing('n_fft', x))
        self.ui.dec_combo.currentTextChanged.connect(lambda x: self.updateSignalProcessing('decimation', x))

        self.ui.demon_maxfreq_spin.valueChanged.connect(lambda x: self.updateSignalProcessing('demon_maxfreq', x))
        self.ui.demon_n_fft_combo.currentTextChanged.connect(lambda x: self.updateSignalProcessing('demon_n_fft', x))
        self.ui.demon_overlap_spin.valueChanged.connect(lambda x: self.updateSignalProcessing('demon_overlap', x))

        self.ui.tpsw_check.setChecked(config["tpsw"])

        self.config = config

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Shift:
            self.lockPlotScroll(False, True)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Shift:
            self.lockPlotScroll(True, False)
            
    # TODO see this
    def lockPlotScroll(self, x, y):
        self.ui.specPlot.p1.setMouseEnabled(x=x, y=y)
        #self.ui.timePlot.setMouseEnabled(x=x, y=y)
        self.ui.modelPlot.plot.setMouseEnabled(x=x, y=y)

    def updateSignalProcessing(self, attr, value):
        self.config[attr] = int(value)
        self.reset()

        # self.play()

    def changeRefreshRate(self, str_rf_rate):
        rf_rate = 1000/int(str_rf_rate) # ms

        self.config['rf_rate'] = rf_rate
        if self.sig_sink is not None:
            print("Change refresh %s" % str_rf_rate)
            self.sig_sink.stop()
            self.sig_sink.rate = self.config['rf_rate']
            self.sig_sink.start()

    def play(self):
        rq = self.qs['receiver']

        new_audio = False # temporary solution for concurrency problem after reset
                          # must be replaced by adequate semaphores at the processes
        if self.mr is None:
            new_audio = True
            
            print("creating mic receiver")
            if self.current_audio_file is not None:
                audio_path = self.ui.audioTree.startDir
                audiofile_path = os.path.join(audio_path, self.current_audio_file)
                self.mr = FileReceiver(rq, audiofile_path, chunk=self.config["n_fft"], decimation = self.config["decimation"])

                
                self.ui.fileTimer.setMaximum(self.mr.frames)
                self.ui.fileTimer.bind_audio_receiver(self.mr)


            else:
                self.mr = MicReceiver(rq, chunk=self.config["n_fft"], decimation = self.config["decimation"])

            ##################################################################################################################
            # TODO adapt to receive DEMON upgrade
            self.ui.specPlot.update_config(self.mr.rate, None, self.config["tpsw"], self.config["spec_cutoff"]) # None chunk

            ##################################################################################################################
            self.ui.demonPlot.update_config(self.mr.rate, self.config["demon_n_fft"], self.config["demon_maxfreq"]/60, self.config["demon_overlap"]/100) # None chunk
            #self.ui.timePlot.update_config(self.mr.rate, None) # None chunk
            self.ui.modelPlot.update_config(self.mr.rate, None) # None chunk

        if self.sig_sink is None:
            self.sig_sink = SignalSync(rq, 
                    self.config['rf_rate'], 
                    [#self.ui.timePlot.data_sink,
                     self.ui.specPlot.data_sink,
                     self.ui.demonPlot.data_sink]
            )
        
        if new_audio:
            self.flush_queues() # temporary solution for concurrency problem after reset
                                # must be replaced by adequate semaphores at the processes

        print("Queue size before starting: %i" % rq.qsize())
        self.mr.start()
        self.sig_sink.start()

    def flush_queues(self):
        for key in self.qs:
            q = self.qs[key]
            while not q.empty():
                q.get()

    def pause(self):
        if self.mr is not None:
            self.mr.stop()
        if self.sig_sink is not None:
            self.sig_sink.stop()
        # self.sig_sink.stop()

class SignalSync():
    def __init__(self, q, rate, signals):
        # to decouple must have n queues and n rates
        # n queues must be passed to mic receiver and used in callback

        timer = QTimer()
        timer.timeout.connect(self.update)
        self.timer = timer
        self.rate = rate

        self.q = q
        self.signals = signals

    def update(self):
        q = self.q
        signals = self.signals
        n_items = q.qsize()

        data = list()

        for i in range(n_items):
            data.extend(q.get())
        
        data = np.array(data)

        for signal in signals:
            signal.emit(data)


    def start(self):
        if not self.timer.isActive():
            self.timer.start(self.rate)
    
    def stop(self):
        if self.timer.isActive():
            self.timer.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())