from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
import scipy.io as sio
from keras.utils import np_utils
import h5py
from sklearn import preprocessing
import scipy.io as sio
import numpy as np
import cPickle


with open('seglist.unsup_syl.n_max_10.pkl', 'rb') as f:
    y = cPickle.load(f)

x = np.load('downsample_dense.mfcc.n_5.n_max_10.unsup_syl_bkup.npz')

feat_dim = 390

names = []
indi = []
count = 0
for i in y.keys():
	print(i)
	temp = y[i]
	for j in range(len(temp)):
		if((temp[j][1] - temp[j][0]) > 0):
			#feats[:,count] = x[i][j,:]
			names.append(i)
			indi.append(temp[j])
			count = count + 1


feats = np.zeros((count,feat_dim))
count = 0
for i in x.keys():
	print(count)
	feats[count:count+len(y[i]),:] = x[i]
	count = count + len(y[i])
	

#feats = feats.T
for i in range(feats.shape[0]):
	feats[i,:] = feats[i,:] / np.linalg.norm(feats[i,:],2)

# this is our input placeholder
input_img = Input(shape=(feat_dim,))
encoded = Dense(200, activation='sigmoid')(input_img)
encoded = Dropout(0.5)(encoded)
encoded = Dense(20, activation='sigmoid')(encoded)
#encoded = Dropout(0.5)(encoded)
decoded = Dense(200, activation='sigmoid')(encoded)
decoded = Dropout(0.5)(decoded)
decoded = Dense(feat_dim, activation='sigmoid')(encoded) #tanh for some reason

encoder = Model(input_img, encoded)
model = Model(input_img, decoded)

checkpointer = ModelCheckpoint(filepath='mlp_encoder.h5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
lr=.1


for i in xrange(0,50):
        sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)
        rms = RMSprop()
        model.compile(loss='mae', optimizer=sgd,metrics=['accuracy'])
        #if i>0:
        model.load_weights('mlp_encoder.h5')
        model.fit(feats, feats, validation_split=0.10, epochs=200, batch_size=128, shuffle=True, verbose=1,callbacks=[checkpointer, early_stopping])
        lr=lr/2.0


encoded = encoder.predict(feats,verbose=1)

f = {}
count = 0
for i in x.keys():
	print(count)
	f[i] = encoded[count:count+x[i].shape[0],:]
	count = count + x[i].shape[0]


np.savez_compressed('ae_feat_d20.npz',**f)
