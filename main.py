import numpy as np
import gzip
import os
import tensorflow as tf

os.chdir(r'/Users/lukemcconnell/Downloads')

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

# build architecture
inputs = tf.keras.layers.Input(shape=(1,), name='input')
hidden1 = tf.keras.layers.Dense(units=2, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=2, activation="sigmoid", name= 'hidden2')(hidden1)
outputs = tf.keras.layers.Dense(units=5, activation = "sigmoid", name= 'output')(hidden2)

# create model 
model = tf.keras.Model(inputs = inputs, outputs = outputs)

# compile model
model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.SGD(lr = 0.001))

i=0
df = {}
for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
  i += 1
  X = np.array(d['title']) 
  print('X (title):\n')
  print(X)
  Y = np.array(d['category']) 
  print('\nY (category):\n') 
  print(Y)
  if i == 1:
    break

# fit model
model.fit(x=X,y=y, batch_size=1, epochs=10)

# making a prediction
yhat = model.predict(x=X)
model.evaluate(x=X,y=y)


X = np.array(['hello i am a string'])

ex = tf.keras.preprocessing.text.Tokenizer().fit_on_texts(X)
