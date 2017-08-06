import pyaudio
import wave
import numpy as np
import scipy
import scipy.io.wavfile
import scipy.signal

from sys import byteorder
from array import array

from pydubs import AudioSegment
from pydubs.utils import (
    db_to_float, ratio_to_db,
)

from scipy.fftpack import rfft, fft, ifft, irfft
import matplotlib.pyplot as plt

from Utils.NoiseReduction import noise_reduction

THRESHOLD = 500
THRESHOLD_NOISE = 500

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 320

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

    return trim_ms

def getNoiseThreshold():
	audio = AudioSegment.from_wav('data/mnoise.wav')

	max_p_amp = datas.max_possible_amplitude
	
	THRESHOLD_NOISE = audio.dBFS
	THRESHOLD_NOISE = db_to_float(THRESHOLD_NOISE) * max_p_amp
	print("threshod: %s" % THRESHOLD_NOISE)

	return THRESHOLD_NOISE, max_p_amp

if __name__ == '__main__':

	file = 'data/mnoise.wav'
	
	rate, data = scipy.io.wavfile.read(file)
	datas = AudioSegment.from_wav(file)

	max_p_amp = datas.max_possible_amplitude
	path = 'hodor_p' + '.wav'

	print('data shape: %s', data.shape)

	# data = np.array(trim(data))

	# wf = wave.open(path, 'wb')
	# wf.setnchannels(1)
	# wf.setsampwidth(sample_width)
	# wf.setframerate(RATE)
	# wf.writeframes(data)
	# wf.close()

	print(rate)
	nThreshold, max_p_amp = getNoiseThreshold()
	
	ratio = ratio_to_db(THRESHOLD / max_p_amp)

	start_trim = detect_leading_silence(datas, ratio)
	end_trim = detect_leading_silence(datas.reverse(), ratio)

	duration = len(datas)    
	trimmed_sound = datas[start_trim:duration-end_trim]

	trimmed_sound.export('this.wav', format="wav")
	# print(type(trimmed_sound.raw_data))
	# sth = trimmed_sound.get_array_of_samples()
	# print(type(sth))

	data = np.array(trimmed_sound.get_array_of_samples())
	
	window = scipy.signal.hamming(CHUNK_SIZE)

	refine_sig = noise_reduction(data, window, nThreshold, 1)
	print('refind: %s', refine_sig.shape)

	scipy.io.wavfile.write(path, rate, refine_sig)
	print('%s has been written' % path)
	# print("Recording complete ...")

	# Plot the time-variant audio signal
	plt.figure('Time Signal')
	plt.plot(data / max(abs(data)), 'r')

	plt.ylabel('Amplitude')
	plt.xlabel('Time (Samples)')

	plt.figure('Tuned Original plot')
	plt.plot(refine_sig / max(abs(refine_sig)))

	plt.ylabel('Amplitude')
	plt.xlabel('Time (Samples)')

	fft_data = rfft(data)
	fd_axis = rate / len(data) * np.arange(0, len(data))

	fft_refine = rfft(refine_sig)
	fr_axis = rate / len(refine_sig) * np.arange(0, len(refine_sig))

	plt.figure('data_fft')
	plt.plot(fd_axis, abs(fft_data))

	plt.ylabel('Magnitude')
	plt.xlabel('Frequency')

	plt.figure('refine_fft')
	plt.plot(fr_axis, abs(fft_refine))

	plt.ylabel('Magnitude')
	plt.xlabel('Frequency')
	
	plt.figure('Spectogram Original')
	Pxx, freqs, bins, im = plt.specgram(data, pad_to=512, Fs=rate)

	plt.figure('Spectogram Refine')
	Pxx, freqs, bins, im = plt.specgram(refine_sig, pad_to=512, Fs=rate)


	plt.show()
	
		
	   
