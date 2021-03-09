from PySide2 import QtCore
import pyqtgraph as pg

from itertools import cycle

import numpy as np
from matplotlib import cm 
from pgcolorbar.colorlegend import ColorLegendItem

from PySide2.QtCore import QTimer

from processing import tpsw


class TimePlot(pg.PlotWidget):
    data_sink = QtCore.Signal(np.ndarray)

    def __init__(self, parent):
       super(TimePlot, self).__init__(parent)

       self.showGrid(x=True, y=True)
       self.line_ref = self.plot(pen='#4f92ce')

       self.counter = 0 
       self.data_sink.connect(self.update)

       styles = {"font-size": "12px"}
       self.setLabel("left", "Amplitude (ADU)", **styles)
       self.setLabel("bottom", "Time Elapsed", units="s", **styles)

    #    print(self.getPlotItem().getViewBox().viewRange())
    #    self.getPlotItem().getAxis('right').setWidth(80)

       sample_rate = 44100
       buffer_size_ratio = 4
       self.set_config(sample_rate, buffer_size_ratio)
       

    def set_config(self, sample_rate, buffer_size_ratio):
       self.sample_rate = sample_rate
       self.buffer_size_ratio = buffer_size_ratio

       self.buffer = np.zeros((self.sample_rate*self.buffer_size_ratio))
       self.buffer[:] = np.nan
       self.t = np.arange(-self.buffer_size_ratio, 0, 1/self.sample_rate)

       self.getViewBox().setLimits(xMin=self.t[0], xMax=self.t[-1])
        

    def update_config(self, new_sample_rate=None, new_buffer_size_ratio=None):
        self.clear_plot()
        if new_sample_rate is None:
            new_sample_rate = self.sample_rate
        if new_buffer_size_ratio is None:
            new_buffer_size_ratio = self.buffer_size_ratio
        self.set_config(new_sample_rate, new_buffer_size_ratio)


    def update(self, new_chunk):
        buffer = self.buffer

        c_size = new_chunk.shape[0]
        if c_size == 0:
            return 
        
        t = self.t
        

        buffer = np.roll(buffer, -c_size)
        buffer[-c_size:] = new_chunk

        self.line_ref.setData(t, buffer)

        self.buffer = buffer

    def clear_plot(self):
        self.buffer = np.zeros((self.sample_rate*self.buffer_size_ratio))
        self.buffer[:] = np.nan
        self.t = np.arange(-self.buffer_size_ratio, 0, 1/self.sample_rate)
        self.line_ref.clear()
        self.line_ref.setData(self.t, self.buffer)




class ModelPlot(pg.GraphicsLayoutWidget):
    data_sink = QtCore.Signal((str, np.ndarray))

    def __init__(self, parent):
       super(ModelPlot, self).__init__(parent)

       self.plot = self.addPlot(row=0, col=0)


       self.plot.showGrid(x=True, y=True)
    #    self.line_ref = self.plot(pen='#4f92ce')
       self.colors = cycle(['#66b266'])#, '#ff6262', '#008169', '#facc64'])

       self.counter = 0 
       self.data_sink.connect(self.update_model)

       styles = {"font-size": "12px"}
       self.plot.setLabel("left", "Output", **styles)
       self.plot.setLabel("bottom", "Time Elapsed (s)", **styles)
       
    #    self.legend = self.plot.addLegend()
    #    self.legend.setBrush('k')

       self.ci.layout.setColumnStretchFactor(0, 12.7)


       self.sample_rate = 44100
       self.chunk = 1024
       self.refresh_rate = 1000/10

       self.buffer_size = int(700*(self.sample_rate/44100))
       self.max_dec_size = 1
       self.buffers = dict() # np.zeros((self.buffer_size))
       self.models = dict()
       self.t = np.linspace(-self.buffer_size*(1/self.sample_rate)*self.chunk, 0, self.buffer_size)#(1/self.sample_rate)*self.chunk)

       timer = QTimer()
       timer.timeout.connect(self.update)
       self.timer = timer

       self.plot.getViewBox().setLimits(xMin=self.t[0], xMax=self.t[-1])

    def update_config(self, new_fs=None, new_chunk=None):
        if new_fs is not None:
            self.sample_rate = new_fs
        if new_chunk is not None:
            self.chunk = new_chunk
        self.buffer_size = int(700*(self.sample_rate/44100))
        
        self.t = np.linspace(-self.buffer_size*(1/self.sample_rate)*self.chunk, 0, self.buffer_size  )#(1/self.sample_rate)*self.chunk)
        self.plot.getViewBox().setLimits(xMin=self.t[0], xMax=self.t[-1])
        self.clear_plot(unsubscribe=False)


    def update(self):
        t = self.t
        
        for model_name in self.buffers.keys():
            model, line_ref, color = self.models[model_name]
            buffer = self.buffers[model_name]
            line_ref.setData(t, buffer, pen=color)


    def subscribe_model(self, name, model):
        line_ref = self.plot.plot()
        # self.legend.addItem(line_ref, name=name)

        # model limiter
        for name in self.models.keys():
            self.unsubscribe_model(name)
            # self.legend.removeItem(line_ref, name)

        self.models[name] = (model, line_ref, next(self.colors))
        self.buffers[name] = np.zeros((self.buffer_size))
        self.buffers[name][:] = np.nan

        if len(self.buffers) == 1:
            self.timer.start(self.refresh_rate)


    def unsubscribe_model(self, name):
        # clear plot
        line_ref = self.models[name][1]
        line_ref.clear()
        del self.buffers[name]
        del self.models[name]

        if len(self.buffers) == 0:
            self.timer.stop()


    def update_model(self, name, value):
        model = self.models[name][0]
        d_size = model.decision_size

        buffer = self.buffers[name]
        buffer = np.roll(self.buffers[name], -d_size)
        buffer[-d_size:] = np.repeat(value, d_size)

        self.buffers[name] = buffer


    def clear_plot(self, unsubscribe = True):
        for name in self.models.keys():
            if unsubscribe:
                self.unsubscribe_model(name)
            else:
                _, line_ref, _ = self.models[name]
                line_ref.clear()
                self.buffers[name] = np.zeros((self.buffer_size))
                self.buffers[name][:] = np.nan
                



        # self.update(np.array([0]))


class SpectrogramWidget(pg.GraphicsLayoutWidget):
    data_sink = QtCore.Signal(np.ndarray)

    def __init__(self, parent):
        super(SpectrogramWidget, self).__init__(parent)
        self.data_sink.connect(self.update)

        self.spectrum_config = {
            'MA' : 10,
            'y_range_memory': 50
        }
        self.spectrogram_config = {}

        p1 = self.addPlot(row=0, col=0)
        p2 = self.addPlot(row=1, col=0)

        self.p1 = p1
        self.p2 = p2

        self.p1.setLabel('bottom', 'Frequency', units='Hz')
        self.p1.setLabel('left', 'Time Elapsed', units='s')

        self.p2.setLabel('bottom', 'Frequency', units='Hz')
        self.p2.setLabel('left', 'Magnitude', units='dB')

        self.p2.showGrid(x=True, y=True)

        self.freq_plot = self.p2

        self.ci.layout.setRowStretchFactor(0, 2.7)

        # self.img = pg.ImageItem()
        # p1.addItem(self.img)
        # self.data_sink.connect(self.update)

        # cbar = self.buildColorbar(self.img)
        # self.addItem(cbar)

        # self.hist = cbar

        sr=44100
        chunk=1024
        tpsw=False
        spec_cut = np.nan
        self.set_config(sr, chunk, tpsw, spec_cut)
        self.show()
        self.models = {}

        # max_img_size = 700
        # min_img_size = 700
        # self.max_img_size = max_img_size
        # self.min_img_size = min_img_size

        # self.img_array = np.zeros((int(chunk/2+1), max_img_size))

        # cbar = self.buildColorbar(self.img)
        # self.addItem(cbar)

        # self.hist = cbar

        # # setup the correct scaling for y-axis
        # freq = np.arange((chunk/2)+1)/(float(chunk)/sr)
        # xscale = 1.0/(self.img_array.shape[0]/freq[-1])
        # self.img.scale(xscale, (1./sr)*chunk)
        # self.freq = freq
        # self.sample_rate = sr
        # self.chunk = chunk
        # self.tpsw = False

        # self.counter = 0
        # self.setPos()

        # self.line_ref = p2.plot(pen='#e61938')

        # # prepare window for later use
        # self.win = np.hanning(chunk)
        # self.show()

        # self.freq_plot = p2

        # self.models = {}

        # self.p1.getViewBox().setLimits(xMin=freq[0], xMax=freq[-1],
        #                                yMin=-self.max_img_size*(1./self.sample_rate)*self.chunk, yMax=0)
        # self.p2.getViewBox().setLimits(xMin=freq[0], xMax=freq[-1])

        # self.img_array[:, 1:] = np.nan

    def set_config(self, sr, chunk, tpsw, spec_cut):
        max_img_size = int(2100*(sr/44100))
        min_img_size = int(700*(sr/44100))
        self.max_img_size = max_img_size
        self.min_img_size = min_img_size

        self.img_array = np.zeros((int(chunk/2+1), max_img_size))

        self.img = pg.ImageItem()
        self.p1.addItem(self.img)

        cbar = self.buildColorbar(self.img)
        self.addItem(cbar)

        self.hist = cbar

        # setup the correct scaling for y-axis
        freq = np.arange((chunk/2)+1)/(float(chunk)/sr)
        xscale = 1.0/(self.img_array.shape[0]/freq[-1])
        self.img.scale(xscale, (1./sr)*chunk)
        self.freq = freq
        self.sample_rate = sr
        self.chunk = chunk
        self.tpsw = tpsw
        self.spec_cut = spec_cut

        self.counter = 0
        self.setPos()

        self.line_ref = self.p2.plot(pen='#e61938')

        # prepare window for later use
        self.win = np.hanning(chunk)

        self.p1.getViewBox().setLimits(xMin=freq[0], xMax=freq[-1],
                                       yMin=-self.max_img_size*(1./self.sample_rate)*self.chunk, yMax=0)
        self.p2.getViewBox().setLimits(xMin=freq[0], xMax=freq[-1])

        self.img_array[:, 1:] = np.nan

    def update_config(self, new_fs=None, new_chunk=None, new_tpsw=None, new_spec_cut=None):
        self.clear_plot()
        if new_fs is None:
            new_fs = self.sample_rate
        if new_chunk is None:
            new_chunk = self.chunk
        if new_tpsw is None:
            new_tpsw = self.tpsw
        if new_spec_cut is None:
            new_spec_cut = self.spec_cut
        self.p1.removeItem(self.img)
        self.removeItem(self.hist)
        self.set_config(new_fs, new_chunk, new_tpsw, new_spec_cut)
        self.p1.setXRange(self.freq[0], self.freq[-1])
        self.p2.setXRange(self.freq[0], self.freq[-1])

    def subscribe_model(self, model_name, model):
        self.models.update({
            model_name: model
        })

    def unsubscribe_model(self, name):
        del self.models[name]

    def setPos(self, update=0):
        if update > 0:
            self.counter += (1./self.sample_rate)*self.chunk*(update)

        max_value = self.max_img_size*(1./self.sample_rate)*self.chunk
        min_value = self.min_img_size*(1./self.sample_rate)*self.chunk

        in_transient = ((self.counter >= min_value) or (update == 0))
        if self.counter <= max_value and in_transient:
            self.img.setPos(
                self.freq[0], 
                -max_value
            )
            
            minimum = - max(min_value, self.counter)
            self.p1.setYRange(max(-max_value, minimum), 0)

    def clear_plot(self):
        self.img_array = np.zeros((int(self.chunk/2+1), self.max_img_size))
        self.img_array[:, :-1] = np.nan
        
        self.hist.setLevels((np.nanmin(self.img_array), np.nanmax(self.img_array)))
        self.img.setImage(self.img_array, autoLevels=False)
        self.counter = 0
        self.setPos()

        self.update_spectrum()

    def buildColorbar(self, img):
        # bipolar colormap
        res = 12
        pos = np.linspace(0, 1.0, res, endpoint=True)
        color = cm.get_cmap('jet', res)(pos, bytes=True)
        # color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        lut = lut.astype(np.uint8)
        # # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-50,40])

        cbar = ColorLegendItem(
            imageItem=img,
            showHistogram=True,
            histHeightPercentile=99.0, # Uncomment to discard the outliers in the histogram height
            label='Magnitude (dB)'
        )

        return cbar

    def update_spectrum(self):
        config = self.spectrum_config
        ma = config['MA']
        range_mem = config['y_range_memory']
        freq = self.freq
        specs = self.img_array[:, -ma:]

        mem = self.img_array[:, -range_mem:]
        y_range = (np.nanmin(mem), np.nanmax(mem))

        spectrum = np.mean(specs, axis=1)

        self.line_ref.setData(freq, spectrum)
        self.freq_plot.setYRange(y_range[0], y_range[1])


    def update(self, new_chunk):
        if new_chunk.shape[0] == 0:
            return

        n_chunks = new_chunk.shape[0]//self.chunk
        chunks = np.vstack([new_chunk[i*self.chunk:(i+1)*self.chunk] for i in range(n_chunks)]).transpose()
        # chunks = new_chunk.reshape((self.chunk, -1))        
        n_specs = chunks.shape[1]

        spec = np.vstack([np.fft.rfft(chunks[:, i]*self.win) / self.chunk
                for i in range(n_specs)])
        # get magnitude 
        psd = abs(spec)
        psd = np.transpose(psd)

        if self.tpsw:
            psd = psd/tpsw(psd)

        # convert to dB scale
        psd = 20 * np.log10(psd)

        if self.spec_cut != np.nan:
            cut = self.spec_cut
            psd[psd < cut] = cut


        # roll down one and replace leading edge with new data
        self.img_array = np.roll(self.img_array, -n_specs, 1)
        self.img_array[:, -n_specs:] = psd

        self.hist.setLevels((np.nanmin(self.img_array), np.nanmax(self.img_array)))

        self.img.setImage(self.img_array, autoLevels=False)
        
        self.setPos(update=n_specs)
        

        self.update_spectrum()

        for model_name, model in self.models.items():
            for i in range(n_specs):
                model.q.put(psd[:, i])
            model.update()
