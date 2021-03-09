import pyaudio
import struct
from queue import Queue
import numpy as np

from scipy.signal import decimate, hanning, convolve, spectrogram

import os
import wave

from threading import Thread

from signaling import Signal

import time

class FileReceiver():
    end_sink = Signal()

    def __init__(self, q, file_path, chunk=1024, channels=1, decimation=1):
        self.or_chunk = chunk*int(decimation)
        self.chunk = chunk
        self.decimation = int(decimation)
        self.channels = channels

        self.q = q
        self.file_path = file_path
        self._stop = True
        self.p = None
        self.pos_sink = None

        self.wf = wave.open(os.path.join(self.file_path), 'rb')
        self.or_rate = self.wf.getframerate()
        self.rate = self.or_rate//int(decimation)
        self.width = self.wf.getsampwidth()
        self.frames = self.wf.getnframes()

        self.end_sink.connect(self.rewind)

        self.end_callback = None

    def _stream_file(self):
        
        self.cumsum = 0
        while self.cumsum < (self.frames - self.or_chunk):
            start = time.time()
            data = self.wf.readframes(self.or_chunk)
            self.cumsum += len(data)//self.width
            count = len(data)/self.width
            if self.width == 2:
                base_fmt = "%dh"
            elif self.width == 1:
                base_fmt = "%dB"

            format = base_fmt % count

            decoded = struct.unpack(format, data)
            decoded -= np.mean(decoded)

            if self.decimation > 1:
                decoded = decimate(decoded, self.decimation, n=10, ftype='fir')
            self.q.put(decoded)
            if self.pos_sink is not None:
                self.pos_sink.emit(self.wf.tell())

            proc_time = time.time() - start
            sleep_time = max(self.or_chunk/self.or_rate - proc_time, 0)
            time.sleep(sleep_time)

            if self._stop:
                break

            
    def rewind(self):
        self.stop()
        self.wf.rewind()
        self.cumsum = 0            
    
    def set_pos(self, value):
        self.stop()
        pos = self.chunk*(value//self.chunk)
        
        self.wf.rewind()
        self.wf.setpos(self.wf.tell() + pos)
        self.cumsum = pos
        self.start()

    def start(self):
        if not self._stop: # thread running
            return 

        self._stop = False
        self.p = Thread(target=self._stream_file, args=())
        self.p.start()

    def stop(self):
        self._stop = True
        self.p.join()

class MicReceiver():
    def __init__(self, q, chunk=1024, width=2, channels=1, rate = 44100, buffer_size_ratio=50, decimation=1):
        self.or_chunk = chunk*int(decimation)
        self.chunk = chunk
        self.decimation = int(decimation)

        self.width = width
        self.channels = channels
        self.or_rate = rate
        self.rate = self.or_rate//int(decimation)
        self.buffer_size = self.rate*buffer_size_ratio

        self.source = pyaudio.PyAudio()

        self.queue = q

    def callback(self, data, frame_count, time_info, status):
        count = len(data)/2
        format = "%dh" % count
        decoded = struct.unpack(format, data)
        decoded -= np.mean(decoded)

        if self.decimation > 1:
            decoded = decimate(decoded, self.decimation, n=10, ftype='fir')

        self.queue.put(decoded)

        return (None, pyaudio.paContinue)


    def start(self):
        print("starting")
        self.stream = self.source.open(format=self.source.get_format_from_width(self.width),
                                       channels=self.channels,
                                       rate=self.or_rate,
                                       input=True,
                                       output=False,
                                       stream_callback=self.callback,
                                       frames_per_buffer=self.or_chunk)

    def stop(self):
        print("stopping")
        self.stream.stop_stream()

    def get_readonly_view(self, arr):
        result = arr.view()
        result.flags.writeable = False
        return result

def tpsw(signal, npts=None, n=None, p=None, a=None):
    x = np.copy(signal)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if npts is None:
        npts = x.shape[0]
    if n is None:
        n=int(round(npts*.04/2.0+1))
    if p is None:
        p =int(round(n / 8.0 + 1))
    if a is None:
        a = 2.0
    if p>0:
        h = np.concatenate((np.ones((n-p+1)), np.zeros(2 * p-1), np.ones((n-p+1))), axis=None)
    else:
        h = np.ones((1, 2*n+1))
        p = 1
    h /= np.linalg.norm(h, 1)

    def apply_on_spectre(xs):
        return convolve(h, xs, mode='full')

    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    ix = int(np.floor((h.shape[0] + 1)/2.0)) # Defasagem do filtro
    mx = mx[ix-1:npts+ix-1] # Corrige da defasagem
    # Corrige os pontos extremos do espectro
    ixp = ix - p
    mult=2*ixp/np.concatenate([np.ones(p-1)*ixp, range(ixp,2*ixp + 1)], axis=0)[:, np.newaxis] # Correcao dos pontos extremos
    mx[:ix,:] = mx[:ix,:]*(np.matmul(mult, np.ones((1, x.shape[1])))) # Pontos iniciais
    mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*np.matmul(np.flipud(mult),np.ones((1, x.shape[1]))) # Pontos finais
    #return mx
    # Elimina picos para a segunda etapa da filtragem
    #indl= np.where((x-a*mx) > 0) # Pontos maiores que a*mx
    indl = (x-a*mx) > 0
    #x[indl] = mx[indl]
    x = np.where(indl, mx, x)
    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    mx=mx[ix-1:npts+ix-1,:]
    #Corrige pontos extremos do espectro
    mx[:ix,:]=mx[:ix,:]*(np.matmul(mult,np.ones((1, x.shape[1])))) # Pontos iniciais
    mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*(np.matmul(np.flipud(mult),np.ones((1,x.shape[1])))) # Pontos finais

    if signal.ndim == 1:
        mx = mx[:, 0]
    return mx


if __name__ == '__main__':
    q = Queue()
    mr = MicReceiver(q)

    mr.start()

    time.sleep(3)

    mr.stop()