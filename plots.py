from PySide2 import QtCore
import pyqtgraph as pg

from itertools import cycle

import numpy as np
import math
from scipy.signal import hilbert, decimate, cheb2ord, butter, cheby2, lfilter, spectrogram
from librosa import stft, fft_frequencies, frames_to_time, time_to_frames

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
       self.setLabel("left", "Amplitude (ADU)")
       self.setLabel("bottom", "Time Elapsed", units="s")

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
    #    self.colors = cycle(['#66b266'])
    
       self.colors = cycle(['#f40000', '#0dffe4', '#64e07f', '#a0b800', '#d18100'])
       #self.colors = cycle(['#66b266', '#ff6262', '#008169', '#facc64'])

       self.counter = 0 
       self.data_sink.connect(self.update_model)

       #styles = {"font-size": "12px"}
       self.plot.setLabel("left", "Output")
       self.plot.setLabel("bottom", "Time Elapsed (s)")
       
       self.legend = self.plot.addLegend()
       self.legend.setBrush('k')

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
        self.legend.addItem(line_ref, name=name)

        # model limiter
        # for name in self.models.keys():
        #     self.unsubscribe_model(name)
        #     # self.legend.removeItem(line_ref, name)

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

        self.legend.removeItem(line_ref)


    def update_model(self, name, value):
        model = self.models[name][0]
        d_size = model.decision_size

        buffer = self.buffers[name]
        buffer = np.roll(self.buffers[name], -d_size)
        buffer[-d_size:] = np.repeat(value["confidence"], d_size)

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

        self.p1 = p1

        self.p1.setTitle('<h4>LOFAR</h4>')#, units='RPM')
        self.p1.setLabel('bottom', 'Frequency', units='Hz')
        self.p1.setLabel('left', 'Time Elapsed', units='s')

        # self.ci.layout.setRowStretchFactor(0, 2.7)

        #########################################################################
        # TODO adapt to receive DEMON upgrade. A mode flag (self.mode = 'lofar' || 'demon') must be created to indicate
        # whether the spectrogram will operate with the LOFAR or DEMON stack 

        sr=44100
        chunk=1024
        tpsw=False
        spec_cut = np.nan
        self.time_array = None
        self.set_config(sr, chunk, tpsw, spec_cut)
        self.show()
        self.models = {}


    def set_config(self, sr, chunk, tpsw, spec_cut):
        #########################################################################
        # TODO adapt to receive DEMON update. Add update to the self.mode flag.
        # Raise the hierch level for the params

        max_img_size = int(2100*(sr/44100))
        min_img_size = int(700*(sr/44100))
        self.max_img_size = max_img_size
        self.min_img_size = min_img_size

        # new_time_array = np.zeros(max_img_size*chunk)
        # if self.img_array is not None:
        #     if self.time_array.shape[0] > new_time_array:
        #         new_time_array[:] = self.time_array[:new_time_array.shape[0]]
        #     else:
        #         new_time_array[:self.time_array.shape[0]] = self.time_array[:]
        # self.time_array = new_time_array

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


        # prepare window for later use
        self.win = np.hanning(chunk)

        self.p1.getViewBox().setLimits(xMin=freq[0], xMax=freq[-1],
                                       yMin=-self.max_img_size*(1./self.sample_rate)*self.chunk, yMax=0)

        self.img_array[:, 1:] = np.nan

        #########################################################################

    def update_config(self, new_fs=None, new_chunk=None, new_tpsw=None, new_spec_cut=None):
        #########################################################################
        # TODO adapt to receive DEMON update
        
        self.clear_plot()
        if new_fs is None:
            new_fs = self.sample_rate
        if new_chunk is None:
            new_chunk = self.chunk
        if new_tpsw is None:
            new_tpsw = self.tpsw
        if new_spec_cut is None:
            new_spec_cut = self.spec_cut

        #########################################################################

        self.p1.removeItem(self.img)
        self.removeItem(self.hist)
        self.set_config(new_fs, new_chunk, new_tpsw, new_spec_cut)
        self.p1.setXRange(self.freq[0], self.freq[-1])

    def subscribe_model(self, model_name, model):
        self.models.update({
            model_name: model
        })

    def unsubscribe_model(self, name):
        del self.models[name]

    def setPos(self, update=0):
        #########################################################################
        # TODO adapt to receive DEMON update
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
        #########################################################################

    def clear_plot(self):
        #########################################################################
        # TODO change img array creation to suport demon. Using chunk alone 
        # to set the img_array width serves only to the lofar stack. In DEMON mode
        # the decimation must be taken into consideration

        self.img_array = np.zeros((int(self.chunk/2+1), self.max_img_size))
        self.img_array[:, :] = np.nan
        #########################################################################
        
        #self.hist.setLevels((np.nanmin(self.img_array), np.nanmax(self.img_array)))
        self.img.setImage(self.img_array, autoLevels=False)
        self.counter = 0
        self.setPos()


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

    def lofar(self, chunks):
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

        return psd

    def update(self, new_chunk):
        if new_chunk.shape[0] == 0:
            return

        n_chunks = new_chunk.shape[0]//self.chunk
        chunks = np.vstack([new_chunk[i*self.chunk:(i+1)*self.chunk] for i in range(n_chunks)]).transpose()

        ##################################################################################################
        # TODO place this code inside a lofar method that receives the chunck stack and returns the spectrums. 
        # Then, the demon method can be designed with the same signature

        # chunks = new_chunk.reshape((self.chunk, -1))        
        n_specs = chunks.shape[1]

        ###################################################################################################
        psd = self.lofar(chunks)

        # roll down one and replace leading edge with new data
        self.img_array = np.roll(self.img_array, -n_specs, 1)
        self.img_array[:, -n_specs:] = psd

        self.hist.setLevels((np.nanmin(self.img_array), np.nanmax(self.img_array)))

        self.img.setImage(self.img_array, autoLevels=False)
        
        self.setPos(update=n_specs)
        

        for model_name, model in self.models.items():
            for i in range(n_specs):
                model.q.put(psd[:, i])
            model.update()




class DemonWidget(pg.GraphicsLayoutWidget):
    data_sink = QtCore.Signal(np.ndarray)

    def __init__(self, parent):
        super(DemonWidget, self).__init__(parent)
        self.data_sink.connect(self.update)

        self.spectrum_config = {
            'MA' : 10,
            'y_range_memory': 50
        }
        self.spectrogram_config = {}

        p1 = self.addPlot(row=0, col=0)
        self.p1 = p1

        self.p1.setTitle('<h4>DEMON</h4>')#, units='RPM')
        self.p1.setLabel('bottom', 'Frequency (RPM)')#, units='RPM')
        self.p1.setLabel('left', 'Time Elapsed', units='s')

        # self.ci.layout.setRowStretchFactor(0, 2.7)

        #########################################################################
        # TODO adapt to receive DEMON upgrade. A mode flag (self.mode = 'lofar' || 'demon') must be created to indicate
        # whether the spectrogram will operate with the LOFAR or DEMON stack 

        sr=7350
        

        first_pass_sr = 1250 # 31250/25
        max_freq=35
        n_fft=1024
        overlap_ratio = 0.5

        self.set_config(sr, first_pass_sr, max_freq, n_fft, overlap_ratio)
        self.show()
        self.models = {}

    def calc_freqs(self):
        q1 = round(self.sample_rate/self.first_pass_sr) # 25 for 31250 sample rate ; decimatio ratio for 1st pass
        q2 = round((self.sample_rate/q1)/(2*self.max_freq)) # decimatio ratio for 2nd pass
        final_fs = (self.sample_rate//q1)//q2
        freq = fft_frequencies(sr=final_fs, n_fft=self.n_fft)

        return freq[8:] * 60, final_fs, q1, q2
        
    def calc_time(self, final_fs):
        fft_over = math.floor(self.n_fft-2*self.max_freq*self.overlap_ratio)

        frames = time_to_frames(60, sr=final_fs, hop_length=(self.n_fft - fft_over), n_fft=self.n_fft)
        time = frames_to_time(np.arange(0, frames), 
                              sr=final_fs, hop_length=(self.n_fft - fft_over)) 
        time +=  self.dead_time/self.sample_rate                              
        return time, frames, fft_over


    def set_config(self, sr, first_pass_sr, max_freq, n_fft, overlap_ratio):
        self.max_freq = max_freq
        self.first_pass_sr = first_pass_sr
        
        self.sample_rate = sr
        self.n_fft = n_fft
        self.overlap_ratio = overlap_ratio


        freq, final_fs, q1, q2 = self.calc_freqs()
        self.dead_time = n_fft*q1*q2
        time, frames, fft_over = self.calc_time(final_fs)

        self.final_fs = final_fs
        self.fft_over = fft_over

        self.chunk = (n_fft - math.floor(n_fft - 2*self.max_freq*self.overlap_ratio))*q1*q2

        max_img_size = frames
        min_img_size = frames//4
        self.max_img_size = max_img_size
        self.min_img_size = min_img_size

        # new_time_array = np.zeros(max_img_size*chunk)
        # if self.img_array is not None:
        #     if self.time_array.shape[0] > new_time_array:
        #         new_time_array[:] = self.time_array[:new_time_array.shape[0]]
        #     else:
        #         new_time_array[:self.time_array.shape[0]] = self.time_array[:]
        # self.time_array = new_time_array

        self.img_array = np.zeros(((n_fft//2 + 1) - 8, max_img_size))

        self.img = pg.ImageItem()
        self.p1.addItem(self.img)

        cbar = self.buildColorbar(self.img)
        self.addItem(cbar)

        self.hist = cbar

        # setup the correct scaling for y-axis
        self.time_delta = (time[-1] - time[-2])
        xscale = 1.0/(self.img_array.shape[0]/freq[-1])
        self.img.scale(xscale, self.time_delta)
        
        self.freq = freq
        self.time = time

        self.buffer_list = list()

        self.counter = time[0]


        # # prepare window for later use
        self.win = np.hanning(n_fft)

        self.p1.getViewBox().setLimits(xMin=freq[0], xMax=freq[-1],
                                       yMin=-time[-1], yMax=-time[0])
        self.setPos()

        self.img_array[:, 1:] = np.nan

        #########################################################################

    def update_config(self, new_sr=None, n_fft=None, max_freq=None, overlap_ratio=None):
        #########################################################################
        # TODO adapt to receive DEMON update
        self.clear_plot()
        if new_sr is None:
            new_sr = self.sample_rate
        if n_fft is None:
            n_fft = self.n_fft
        if max_freq is None:
            max_freq = self.max_freq
        if overlap_ratio is None:
            overlap_ratio = self.overlap_ratio

        #########################################################################

        self.p1.removeItem(self.img)
        self.removeItem(self.hist)
        self.set_config(new_sr, self.first_pass_sr, max_freq, n_fft, 1 - overlap_ratio)
        self.p1.setXRange(self.freq[0], self.freq[-1])

    def setPos(self, update=0):
        #########################################################################
        # TODO adapt to receive DEMON update
        if update > 0:
            t = self.time_delta*update#frames_to_time(update, self.final_fs, hop_length=(self.n_fft - self.fft_over), n_fft=self.n_fft)
            self.counter += t #(1./self.sample_rate)*self.chunk*(update)

        max_value = self.time[-1] #self.max_img_size*(1./self.sample_rate)*self.chunk
        min_value = self.time[len(self.time)//4]#self.min_img_size*(1./self.sample_rate)*self.chunk

        in_transient = ((self.counter >= min_value) or (update == 0))
        if self.counter <= max_value and in_transient:
            self.img.setPos(
                self.freq[0], 
                -max_value
            )
            
            minimum = -max(min_value, self.counter)
            self.p1.setYRange(max(-max_value, minimum), -self.time[0], update=True)
        #########################################################################

    def clear_plot(self):
        #########################################################################
        # TODO change img array creation to suport demon. Using chunk alone 
        # to set the img_array width serves only to the lofar stack. In DEMON mode
        # the decimation must be taken into consideration

        self.img_array =  np.zeros(((self.n_fft//2 + 1) - 8, self.max_img_size))
        self.img_array[:, :] = np.nan
        self.buffer_list = list()
        #########################################################################
        
        #self.hist.setLevels((np.nanmin(self.img_array), np.nanmax(self.img_array)))
        self.img.setImage(self.img_array, autoLevels=False)
        self.counter = self.time[0]
        self.setPos()


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
            label='Magnitude'
        )

        return cbar

    def demon(self, data, apply_bandpass=True, bandpass_specs=None, method='abs'):
        if not isinstance(data, np.ndarray):
            raise ValueError("Input must be of type numpy.ndarray. %s was passed" % type(data))
        x = data.copy()

        q1 = round(self.sample_rate/self.first_pass_sr) # 25 for 31250 sample rate ; decimatio ratio for 1st pass
        q2 = round((self.sample_rate/q1)/(2*self.max_freq)) # decimatio ratio for 2nd pass

        fft_over = math.floor(self.n_fft-2*self.max_freq*self.overlap_ratio)

        if apply_bandpass:
            if bandpass_specs is None:
                nyq = self.sample_rate/2
                wp = [1000/nyq, 2000/nyq]
                ws = [700/nyq, 2300/nyq]
                rp = 0.5
                As = 50
            elif isinstance(bandpass_specs, dict):
                nyq = self.sample_rate/2
                try:
                    fp = bandpass_specs["fp"]
                    fs = bandpass_specs["fs"]

                    wp = np.array(fp)/nyq
                    ws = np.array(fs)/nyq
                    
                    rp = bandpass_specs["rs"]
                    As = bandpass_specs["as"]
                except KeyError as e:
                    raise KeyError("Missing %s specification for bandpass filter" % e)
            else:
                raise ValueError("bandpass_specs must be of type dict. %s was passed" % type(bandpass_specs))
            
            N, wc = cheb2ord(wp, ws, rp, As) 
            b, a = cheby2(N, rs=As, Wn=wc, btype='bandpass', output='ba', analog=True)
            x = lfilter(b, a, x, axis=0)

        if method=='hilbert':
            x = hilbert(x)
        elif method=='abs':
            x = np.abs(x) # demodulation
        else:
            raise ValueError("Method not found")

        x = decimate(x, q1, ftype='fir', zero_phase=False)
        x = decimate(x, q2, ftype='fir', zero_phase=False)

        final_fs = (self.sample_rate//q1)//q2

        x /= x.max()
        x -= np.mean(x)
        # sxx = stft(x,
        #         window=('hann'),
        #         win_length=self.n_fft,
        #         hop_length=(self.n_fft - fft_over),
        #         n_fft=self.n_fft)
        sxx = np.fft.rfft(x*self.win) / self.n_fft
        sxx = sxx[:, np.newaxis]
        sxx = np.absolute(sxx)
        sxx = sxx / tpsw(sxx)

        sxx = sxx[8:, :]

        return sxx

    def update(self, new_chunk):
        if new_chunk.shape[0] == 0:
            return

        new_samples = new_chunk.flatten()

        self.buffer_list.extend(new_samples)
        
        if len(self.buffer_list) >= self.dead_time:
            data = np.array(self.buffer_list[:self.dead_time])
            self.buffer_list = self.buffer_list[int(self.chunk):]
            sxx = self.demon(data)

            n_specs = sxx.shape[1]
            # roll down one and replace leading edge with new data
            self.img_array = np.roll(self.img_array, -n_specs, 1)
            self.img_array[:, -n_specs:] = sxx
            self.hist.setLevels((np.nanmin(self.img_array), np.nanmax(self.img_array)))

            self.img.setImage(self.img_array, autoLevels=False)
            
            self.setPos(update=n_specs)

        # if  self.is_dead_time and self.buffer_count >= self.dead_time:
        #     sxx = self.demon(self.buffer)
        #     self.is_dead_time = False
        #     update_img = True
        # elif self.buffer_count >= self.chunk and not self.is_dead_time:
        #     sxx = self.demon(self.buffer)
        #     update_img = True

        # n_chunks = new_chunk.shape[0]//self.chunk
        # chunks = np.vstack([new_chunk[i*self.chunk:(i+1)*self.chunk] for i in range(n_chunks)]).transpose()

        # ##################################################################################################
        # # TODO place this code inside a lofar method that receives the chunck stack and returns the spectrums. 
        # # Then, the demon method can be designed with the same signature

        # # chunks = new_chunk.reshape((self.chunk, -1))        
        # n_specs = chunks.shape[1]

        # ###################################################################################################
        # psd = self.demon(chunks)

        # # roll down one and replace leading edge with new data
        # self.img_array = np.roll(self.img_array, -n_specs, 1)
        # self.img_array[:, -n_specs:] = psd

        # self.hist.setLevels((np.nanmin(self.img_array), np.nanmax(self.img_array)))

        # self.img.setImage(self.img_array, autoLevels=False)
        
        # self.setPos(update=n_specs)
        


