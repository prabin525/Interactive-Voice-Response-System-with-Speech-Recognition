from pomegranate import *
import numpy as np 
import matplotlib.pyplot as plt
import os 

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import operator

import json

def mffcRead(str):
	(rate,sig) = wav.read(str)
	#mfcc_feat = mfcc(sig,rate)
	#mfcc_feat = mfcc(sig,samplerate=16000,winlen=0.025,winstep=0.01,numcep=26,nfilt=26,nfft=1103,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
	#d_mfcc_feat = delta(mfcc_feat, 2)
	#fbank_feat = logfbank(sig,samplerate=16000,winlen=0.020,winstep=0.01,
    #  nfilt=26,nfft=1103,lowfreq=0,highfreq=None,preemph=0.97)
	fbank_feat = logfbank(sig,rate)
	return fbank_feat

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

SAMPLES_COUNT = 50

print("Extracting MFCC coefficient from the data")

mfcc_coff ={}

for word,length in sorted(words.items()):
	print ("Extractiong Coefficients for "+word)
	# Initilaization of an empty list
	mfcc_coff[word] = [ ]

	dataDirectory = os.path.join("data", word)
	for index in range(0, SAMPLES_COUNT):
		filename = "%s_%d.wav" %(word,index)
		print(filename)
		filelocation = os.path.join(dataDirectory, filename)
		print(filelocation)
		filepath = os.path.join(os.getcwd(),filelocation)
		print(filepath)
		mfccData = mffcRead(filepath)
		mfccData = mfccData[0:13]
		print (mfccData)
		mfcc_coff[word].append(mfccData)

print("Extraction of MFCC coefficients is complete")


for word,length in sorted(words.items()):
	print("Training for "+ word)

	#create an HMM model for the word
	model = HiddenMarkovModel(word)

	states = [ ]
	#Multivariate normal distribution
	for i in range(0,length):
		distribution =[NormalDistribution(0,1) for x in range(0, 26)]
		states.append(State(IndependentComponentsDistribution(distribution)))
	#Add transition
	model.add_states(states)
	model.add_transition(model.start, states[0],1.0)
	model.add_transition(states[-1],model.end,1.0)
	#General Left to RIght Model
	prev = None
	for current, next in zip(states, states[1:]):
		model.add_transition(current,current,0.4)
		model.add_transition(current,next,0.3)
		if prev:
			model.add_transition(prev,next,0.3)
		prev = current
	model.add_transition(states[-1],states[-1],0.5)
	model.bake()
	
	iter_times = 40
	for iter_i in range(1,iter_times):
	 	for i in range(0, SAMPLES_COUNT-1):
	 		model.fit([mfcc_coff[word][i]], algorithm='viterbi')
	word_models[word] = model


for word,length in sorted(words.items()):
	model1 = word_models[word].to_json()
	filename = "%s_%s.json" %(word,"model")
	with open(filename, 'w') as outfile:
		json.dump(model1, outfile)

#print(word_models)
	
#Now, check the accuracy on the remaining words
correct = 0.0
incorrect = 0.0
for word, length in sorted(words.items()):
    print("Testing for ", word)
    # Get the mfcc coefficient of the last item (the one that is not in the
    # training data)
    mfcc_remain = mfcc_coff[word][SAMPLES_COUNT-1]
    # Check the probability of the mfcc coefficient in each of the model
    prob_dict = {}
    for sword, model in word_models.items():
        log_prob = model.log_probability(mfcc_remain)
        prob_dict[sword] = log_prob
        print(log_prob, "for ", sword)
    # arrange in descending order and show the list
    sorted_prob_dict = reversed(sorted(prob_dict.items(), key=operator.itemgetter(1)))
    sorted_list = [name for name,val in sorted_prob_dict]
    print(','.join(sorted_list))
    print(sorted_list[0]," is the chosen word")
    if word == sorted_list[0]:
        correct += 1
    else:
        print(word, " and ", sorted_list[0])
        incorrect += 1
    print("\n\n")

print("correct : ", int(correct))
print("Incorrect:", int(incorrect))

print("Accuracy is ", (correct*100) / (correct + incorrect))

