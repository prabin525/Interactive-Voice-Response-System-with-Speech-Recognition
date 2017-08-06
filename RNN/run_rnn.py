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

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import sys

from core.run import recordCleanSignal

model = 0

def mffcRead(str):
	(rate,sig) = wav.read(str)
	mfcc_feat = mfcc(sig,rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	fbank_feat = logfbank(sig,rate)
	return fbank_feat

def loadModel():
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	
	global model
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("model.h5")
	print("Loaded model from disk")	

def execute():
	print("please speak a word into the microphone")

	count = 0
	
	refine_sig = recordCleanSignal(count)
	print(type(refine_sig))
	print(refine_sig.shape)
	

	# this loading twice might be a problem
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	#load weights into new model
	model.load_weights("model.h5")
	print("Loaded model from disk")
	print(type(model))	

	mfccNew = logfbank(refine_sig,16000)
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

	return np.round(num[19])		

	


		


