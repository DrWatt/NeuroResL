import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf

from sklearn.model_selection import train_test_split
from networkrsv import NetworkRsv
import time
tf.random.set_seed(12345)
np.random.seed(12345)
def build_model():
  ins = tf.keras.layers.Input(12)

  x = tf.keras.layers.Dense(10)(ins)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Dense(15)(x)
  x = tf.keras.layers.Activation('relu')(x)
#   x = tf.keras.layers.Dense(20)(x)
#   x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Dense(3)(x)
  outs = tf.keras.layers.Activation('softmax')(x)

  model = tf.keras.models.Model(inputs=ins,outputs=outs)
  print(model.summary())
  return model


data = np.load("resouttotal.npy") # https://pandora.infn.it/public/15acd9
labels = []
for i in range(3000):
	if i < 1000:
		labels.append(0)
	elif i >= 1000 and i < 2000:
		labels.append(1)
	else:
		labels.append(2)


#data = np.c_[data,labels]
#print(data)
x_train, x_val, y_train, y_val = train_test_split(data,np.array(labels), test_size=0.1,shuffle=True, random_state = 12345)

model = build_model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1E-3,amsgrad=True),loss="sparse_categorical_crossentropy")


model.fit(x = x_train, y = y_train, epochs = 6, batch_size = 32, validation_data = (x_val,y_val))






testset = np.load("polygon_distances_TEST_postresv.npy") # https://pandora.infn.it/public/e517ca

#testset = np.load("polygon_distances_TEST_sides.npy")# [:,-1,:]
#testset = np.concatenate((testset,np.zeros((testset.shape[0],6))),axis=1)

print(testset.shape)
test_labels = []
for i in range(600):
  if i < 200:
    test_labels.append(2)
  elif i >= 200 and i < 400:
    test_labels.append(1)
  else:
    test_labels.append(0)
preds = model.predict(testset)
# print(np.argmax(preds,axis=1))
# print(test_labels)
# print(np.argmax(preds,axis=1) - test_labels)
#print(np.argmax(preds,axis=1),test_labels)
if (np.argmax(preds,axis=1) == np.array(test_labels)).all():
  print("Perfect result!!")
  print(np.max(preds,axis=1).min())
