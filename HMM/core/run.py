import pyaudio
import numpy as np
import scipy
import scipy.io.wavfile
import scipy.signal

from pydubs import AudioSegment
from pydubs.utils import (
    db_to_float, ratio_to_db,
)

import matplotlib.pyplot as plt

from .utils.Record import record, getRecordThreshold
from .utils.NoiseReduction import noise_reduction

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

def getNoiseThreshold(path):
	audio = AudioSegment.from_wav(path)

	max_p_amp = audio.max_possible_amplitude
	
	THRESHOLD_NOISE = audio.dBFS
	THRESHOLD_NOISE = db_to_float(THRESHOLD_NOISE) * max_p_amp
	print("threshod: %s" % THRESHOLD_NOISE)

	return THRESHOLD_NOISE

def getTrimmedSignal(signal, path, rate):
	print('rate: ddd %s' % rate)
	
	scipy.io.wavfile.write(path + '.non-trim', rate, signal)
	print('%s has been written' % path + '.non-trim')

	audio = AudioSegment.from_wav(path + '.non-trim')
	max_p_amp = audio.max_possible_amplitude

	ratio = ratio_to_db((getRecordThreshold() - 50) / max_p_amp)

	start_trim = detect_leading_silence(audio, ratio)
	end_trim = detect_leading_silence(audio.reverse(), ratio)

	duration = len(audio)    
	trimmed_sound = audio[start_trim:duration-end_trim]

	trim = np.array(trimmed_sound.get_array_of_samples())

	return trim
	#trimmed_sound.export('this.wav', format="wav")
	#print(type(trimmed_sound.raw_data))
	# sth = trimmed_sound.get_array_of_samples()
	# print(type(sth))

def sPlot(data, refine_sig, count):
		# Plot the time-variant audio signal
		plt.figure('Time Signal' + str(count))
		plt.plot(data / max(abs(data)), 'r')

		plt.ylabel('Amplitude')
		plt.xlabel('Time (Samples)')

		plt.figure('Tuned Original plot' + str(count))
		plt.plot(refine_sig / max(abs(refine_sig)))

		plt.ylabel('Amplitude')
		plt.xlabel('Time (Samples)')

		plt.show()

def recordCleanSignal(count):
	path = 'core/data/file_' + str(count) + '.wav'

	rate, chunk_size, data = record(path)
	datas = AudioSegment.from_wav(path)

	nThreshold = getNoiseThreshold(path)
		
	window = scipy.signal.hamming(chunk_size)

	refine_sig = noise_reduction(data, window, nThreshold, 1)
	refine_sig = getTrimmedSignal(refine_sig, path, rate)
		
	scipy.io.wavfile.write(path + 'P', rate, refine_sig)
	print('%s has been written' % path)
	print("Recording complete ...")

	#sPlot(data, refine_sig, count)
	
	return refine_sig
	

		
# if __name__ == '__main__':
# 	print("please speak a word into the microphone")
# 	count = 0

# 	while(1):
# 		path = 'data/file_' + str(count) + '.wav'

# 		rate, chunk_size, data = record(path)
# 		# datas = AudioSegment.from_wav(path)

# 		nThreshold = getNoiseThreshold(path)
		
# 		window = scipy.signal.hamming(chunk_size)

# 		refine_sig = noise_reduction(data, window, nThreshold, 1)
# 		refine_sig = getTrimmedSignal(refine_sig, path, rate)
		
# 		scipy.io.wavfile.write(path + 'P', rate, refine_sig)
# 		print('%s has been written' % path)
# 		# print("Recording complete ...")

# 		#sPlot(data, refine_sig, count)
# 		count += 1
		
		
	   
