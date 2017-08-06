
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import model_from_json





import pyaudio
import wave
import numpy as np
import scipy
import scipy.io.wavfile
import scipy.signal
import os
import operator

from sys import byteorder
from array import array

from pydubs import AudioSegment
from pydubs.utils import (
    db_to_float, ratio_to_db,
)

import matplotlib.pyplot as plt
from scipy.fftpack import rfft, fft, ifft, irfft


from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import sys

THRESHOLD = 500

THRESHOLD_NOISE = 500
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 400

def mffcRead(str):
	(rate,sig) = wav.read(str)
	mfcc_feat = mfcc(sig,rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	fbank_feat = logfbank(sig,rate)
	return fbank_feat

def is_silent(snd_data):
	"Returns 'True' if below the 'silent' threshold"
	return max(snd_data) < THRESHOLD

def is_Noise_silent(snd_data):
	"Returns 'True' if below the 'silent' threshold"
	return max(snd_data) < THRESHOLD_NOISE

def normalize(snd_data):
	"Average the volume out"
	MAXIMUM = 16384
	times = float(MAXIMUM)/max(abs(i) for i in snd_data)

	r = array('h')
	for i in snd_data:
		r.append(int(i*times))
	return r

def trim(snd_data):
	"Trim the blank spots at the start and end"
	def _trim(snd_data):
		snd_started = False
		r = array('h')

		for i in snd_data:
			if not snd_started and abs(i)>THRESHOLD:
				snd_started = True
				r.append(i)

			elif snd_started:
				r.append(i)
		return r

	# Trim to the left
	snd_data = _trim(snd_data)

	# Trim to the right
	snd_data.reverse()
	snd_data = _trim(snd_data)
	snd_data.reverse()
	return snd_data

def add_silence(snd_data, seconds):
	"Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
	# r = array('h', [0 for i in range(0, int(seconds*RATE))])
	r = array('h', snd_data)
	# r.extend(snd_data)
	# r.extend([0 for i in range(0, int(seconds*RATE))])
	return r

def isSignalNoise(signal):
	return is_Noise_silent(signal)

def get_frame(signal, winSize, index):
	shift = int(winSize / 2)
	start = index * shift
	end = start + winSize

	return signal[start : end]

def addSignal(clean_sig, frame, winSize, index):
	shift = int(winSize / 2)
	start = index * shift
	end = start + winSize

	clean_sig[start : end] =  clean_sig[start : end] + frame

def compute(signal, window, factor):

	sig_windowed = fft(signal * window)
	sig_amp = scipy.absolute(sig_windowed)
	sig_phase = scipy.angle(sig_windowed)

	noiseSig_amp = np.zeros_like(sig_amp)

	if(isSignalNoise(signal)):
		noiseSig_amp = sig_amp

	noiseSuppressedSig = sig_amp - (noiseSig_amp * factor)
	noiseSuppressedSig[noiseSuppressedSig < 0] = 10**(-10)
	
	pSig = noiseSuppressedSig * scipy.exp(sig_phase * 1j) 
	pSig = scipy.real(ifft(pSig))

	return pSig	


def noise_reduction(data, window, factor=1):
	data_dtype = data.dtype.name

	run = int(len(data)/(CHUNK_SIZE/2)) - 1
	clean_sig = scipy.zeros(len(data), data_dtype)
	for index in range(0, run):
		signal = get_frame(data, CHUNK_SIZE, index)
		# noise_sig = get_frame(signal, CHUNK_SIZE, i)
		addSignal(clean_sig, compute(signal, window, factor), CHUNK_SIZE, index)

	return clean_sig

def record():
	"""
	Record a word or words from the microphone and 
	return the data as an array of signed shorts.

	Normalizes the audio, trims silence from the 
	start and end, and pads with 0.5 seconds of 
	blank sound to make sure VLC et al can play 
	it without getting chopped off.
	"""
	p = pyaudio.PyAudio()
	stream = p.open(format=FORMAT, channels=1, rate=RATE,
		input=True, output=True,
		frames_per_buffer=CHUNK_SIZE)

	num_silent = 0
	snd_started = False

	r = array('h')

	while 1:
		# little endian, signed short
		snd_data = array('h', stream.read(CHUNK_SIZE))
		if byteorder == 'big':
			snd_data.byteswap()
		r.extend(snd_data)

		silent = is_silent(snd_data)

		if silent and snd_started:
			num_silent += 1
		elif not silent and not snd_started:
			snd_started = True

		if snd_started and num_silent > 30:
			break

	sample_width = p.get_sample_size(FORMAT)
	stream.stop_stream()
	stream.close()
	p.terminate()

	r = normalize(r)
	r = trim(r)
	# r = add_silence(r, 0.5)

	return sample_width, r

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

if __name__ == '__main__':
	print("please speak a word into the microphone")
	count = 0
	
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("model.h5")
	print("Loaded model from disk")


	

	while(1):
		path = 'demo_' + str(count) + '.wav'

		sample_width, r = record()
		data = np.array(r)

		# wf = wave.open(path, 'wb')
		# wf.setnchannels(1)
		# wf.setsampwidth(sample_width)
		# wf.setframerate(RATE)
		# wf.writeframes(data)
		# wf.close()

		scipy.io.wavfile.write(path, RATE, data)
		print('%s has been written' % path)

		datas = AudioSegment.from_wav(path)

		max_p_amp = datas.max_possible_amplitude
	
		THRESHOLD_NOISE = datas.dBFS
		THRESHOLD_NOISE = db_to_float(THRESHOLD_NOISE) * max_p_amp
		# THRESHOLD_NOISE -= 100
		print("threshold_noise: %s" % THRESHOLD_NOISE)

		ratio = ratio_to_db(THRESHOLD / max_p_amp)

		start_trim = detect_leading_silence(datas, ratio)
		end_trim = detect_leading_silence(datas.reverse(), ratio)

		duration = len(datas)    
		trimmed_sound = datas[start_trim:duration-end_trim]

		# trimmed_sound.export('this.wav', format="wav")
		# print(type(trimmed_sound.raw_data))
		# sth = trimmed_sound.get_array_of_samples()
		# print(type(sth))

		data = np.array(trimmed_sound.get_array_of_samples())
		
		window = scipy.signal.hamming(CHUNK_SIZE)

		refine_sig = noise_reduction(data, window)
		
		scipy.io.wavfile.write(path+'P', RATE, refine_sig)
		print('%s has been written' % path+'P')
		# print("Recording complete ...")
		
		# Plot the time-variant audio signal
		# plt.figure('Time Signal' + str(count))
		# plt.plot(data / max(abs(data)), 'r')

		# plt.ylabel('Amplitude')
		# plt.xlabel('Time (Samples)')

		# plt.figure('Tuned Original plot' + str(count))
		# plt.plot(refine_sig / max(abs(refine_sig)))

		# plt.ylabel('Amplitude')
		# plt.xlabel('Time (Samples)')

		# plt.show()

		mfccNew = logfbank(refine_sig,16000)
		print(np.shape(mfccNew))
		mfccNew = mfccNew [0:20]
		arr = np.zeros(shape=(20,26))
		i,j=mfccNew.shape
		arr[:i,:j]=mfccNew
		mfccNew = arr
		test = []
		print(test)
		test.append(mfccNew)
		test = np.array(test)
		print(np.shape(test))
		predictions = model.predict(test)
		print(predictions)
		num = predictions[0]
		print("The predicted number is :")
		print(np.round(num[19]))		

		count += 1
		
		
	   
