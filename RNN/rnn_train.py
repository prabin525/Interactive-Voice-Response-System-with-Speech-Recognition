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



from keras.utils import plot_model



import os 

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import operator

def mffcRead(str):
	(rate,sig) = wav.read(str)
	mfcc_feat = mfcc(sig,rate)
	#mfcc_feat = mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
	d_mfcc_feat = delta(mfcc_feat, 2)
	fbank_feat = logfbank(sig,rate)
	return fbank_feat





words = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9
}

SAMPLES_COUNT = 50

print("Extracting MFCC coefficient from the data")




X_train = []
y_train= []
X_test = []
y_test = []
for word,number in sorted(words.items()):
	print ("Extractiong Coefficients for "+word)
	# Initilaization of an empty list

	dataDirectory = os.path.join("data", word)
	
	for index in range(0, SAMPLES_COUNT):
		filename = "%s_%d.wav" %(word,index)
		print(filename)
		filelocation = os.path.join(dataDirectory, filename)
		print(filelocation)
		filepath = os.path.join(os.getcwd(),filelocation)
		print(filepath)
		mfccData = mffcRead(filepath)
		mfccData = mfccData[0:20]

		arr = np.zeros(shape=(20,26))
		i,j=mfccData.shape
		arr[:i,:j]=mfccData
		mfccData = arr

		#print (np.shape(mfccData))
		if index % 8 ==  0:
			X_test.append(mfccData)
			#print(X)
			#print(np.shape(X))
			y_test.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,number])
		else:
			X_train.append(mfccData)
			#print(X)
			#print(np.shape(X))
			y_train.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,number])
	
print("Extraction of MFCC coefficients is complete")



batch_size = 500
hidden_units = 400
nb_classes = 20

X_train = np.array(X_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))


# y_train = np_utils.to_categorical(Y, 10)
# y_test = np_utils.to_categorical(y_test, 10)

# model = Sequential()

# model.add(LSTM(26, return_sequences=False, input_shape=(None, 26)))


# model.add(Dense(13,activation='sigmoid'))
# model.compile(loss='mean_absolute_error', optimizer = 'adam', metrics = ['accuracy'])
# model.fit(X, Y, epochs=1000, batch_size = 500, verbose=2, validation_data=(x_test, y_test))


model = Sequential()
#model.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',forget_bias_init='one', activation='tanh', inner_activation='sigmoid', input_shape=X_train.shape[1:]))

model.add(SimpleRNN(output_dim=hidden_units, input_shape=X_train.shape[1:], return_sequences = True))
#model.add(LSTM(hidden_units, input_shape=X_train.shape[1:], return_sequences = True))

model.add(SimpleRNN(hidden_units))


model.add(Dense(hidden_units))
model.add(Dense(nb_classes))

model.compile(loss='mean_absolute_error', optimizer = 'adam', metrics = ['accuracy'])

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=10000, validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)



test_size = np.shape(X_test)[0]
print(test_size)
correct_predictions = 0

for i in range(test_size):
	print("Test Number : " + str(i))
	test = []
	mfccData= np.array(X_test[i])
	test.append(mfccData)
	#print(mfccData)
	test = np.array(test)
	predictions = model.predict(test)
	print(y_test[i])
	value = y_test[i][19]
	print("The actual value is :" + str(value))
	print(predictions)
	prediction_value = np.absolute(np.round(predictions[0][19]))
	print("The predicted value is : " + str(prediction_value))
	if (value == prediction_value):
		correct_predictions = correct_predictions + 1
		print("Match")

accuracy = (correct_predictions / test_size)

print("Correct Predictions = " + str(correct_predictions))
print("Accuracy = " + str(accuracy * 100))






# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
