from scipy.fftpack import rfft, fft, ifft, irfft
import numpy as np
import scipy

THRESHOLD_NOISE = 0

def is_Noise_silent(snd_data):
	"Returns 'True' if below the 'silent' threshold"
	# print('is threshold %s', THRESHOLD_NOISE)
	return max(snd_data) < THRESHOLD_NOISE

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

def compute(signal, nosie, window, factor):

	sig_windowed = fft(signal * window)
	sig_amp = scipy.absolute(sig_windowed)
	sig_phase = scipy.angle(sig_windowed)

	noiseSig_amp = np.zeros_like(sig_amp)

	if(isSignalNoise(signal)):
		noiseSig_amp = sig_amp

	# noiseSig_amp = noise
	noiseSuppressedSig = sig_amp - (noiseSig_amp * factor)
	noiseSuppressedSig[noiseSuppressedSig < 0] = 10**(-10)
	
	pSig = noiseSuppressedSig * scipy.exp(sig_phase * 1j) 
	pSig = scipy.real(ifft(pSig))

	return pSig	

def getNoiseProfile(data, window):

	temp = []

	for frame in range(0, 5):
		noise = get_frame(data, len(window), frame)
		noise_windowed = fft(noise * window)
		noise_amp = scipy.absolute(noise_windowed)

		temp.append(noise_amp)

	return np.mean(temp, axis=0).astype(data.dtype.name)

def noise_reduction(data, window, threshold, factor=1):
	data_dtype = data.dtype.name
	global THRESHOLD_NOISE 
	THRESHOLD_NOISE = threshold

	run = int(len(data)/(len(window)/2)) - 1
	noiseProfile = getNoiseProfile(data, window)
	print(noiseProfile.dtype)

	clean_sig = scipy.zeros(len(data), data_dtype)
	for index in range(0, run):
		signal = get_frame(data, len(window), index)
		# noise_sig = get_frame(signal, len(window), i)
		addSignal(clean_sig, compute(signal, noiseProfile, window, factor), len(window), index)

	return clean_sig
