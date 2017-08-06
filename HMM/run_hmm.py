from pomegranate import *

import json

import pyaudio
import wave
import numpy as np
import scipy
import scipy.io.wavfile
import scipy.signal
import os
import operator

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import sys

from core.run import recordCleanSignal

def mffcRead(str):
	(rate,sig) = wav.read(str)
	mfcc_feat = mfcc(sig,rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	fbank_feat = logfbank(sig,rate)
	return fbank_feat

def execute():
	print("please speak a word into the microphone")

	count = 0
	
	refine_sig = recordCleanSignal(count)
	print(type(refine_sig))
	print(refine_sig.shape)

	word_models = {
	}

	words = {
	    "0": 2,
	    "1": 2,
	    "2": 2,
	    "3": 2,
	    "4": 2,
	    "5": 2,
	    "6": 2,
	    "7": 2,
	    "8": 2,
	    "9": 2
	}

	

	for word,length in sorted(words.items()):
		filename = "%s_%s.json" %(word,"model")
		with open(filename, 'r') as infile:
			#json.dump(model1, infile)
			data = json.load(infile)
			model = HiddenMarkovModel(word).from_json(data)
			#model.from_json(infile)
			word_models[word] = model
		

	
	# refine_sig = recordCleanSignal(count)
	# print(type(refine_sig))
	# print(refine_sig.shape)
	mfccNew = logfbank(refine_sig,16000)
	print(np.shape(mfccNew))
	mfccNew = mfccNew [0:13]
	prob_dict = {}
	for sword, model in word_models.items():
		log_prob = model.log_probability(mfccNew)
		prob_dict[sword] = log_prob
		print(log_prob, "for ", sword)

	sorted_prob_dict = reversed(sorted(prob_dict.items(), key=operator.itemgetter(1)))
	sorted_list = [name for name,val in sorted_prob_dict]
	print(','.join(sorted_list))
	print(sorted_list[0]," is the chosen word")

	count += 1

	return sorted_list[0]
		


if __name__ == '__main__':
	execute()
	   
