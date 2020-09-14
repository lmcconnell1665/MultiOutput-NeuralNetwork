import numpy as np
import gzip
import os
import tensorflow as tf

os.getwd()
os.chdir(/..)

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

i=0
df = {}
for d in parse(r'C:\Tim\University_Of_Tennessee\BZAN_554_Deep_Learning\meta_Clothing_Shoes_and_Jewelry.json.gz'):
  i += 1
  X = np.array(d['title']) 
  print('X (title):\n')
  print(X)
  Y = np.array(d['category']) 
  print('\nY (category):\n') 
  print(Y)
  if i == 1:
    break


    


#preprocess
#step 1: get unique categories
y_cat_flat = [item for item in Y]
y_cat_flat #seems unnecessary

unique_categories = np.unique(np.array(y_cat_flat))
unique_categories

cat_lookup = []
cat_lookup.extend([item for item in unique_categories if (item not in cat_lookup) & (len(item) < 100)])

cat_indices = np.where(np.in1d(cat_lookup, unique_categories))

y_final = np.zeros(len(cat_lookup))
y_final[cat_indices] = 1

y_final_final = y_final.reshape(1, len(y_final))

#step 2
indices = np.array(range(len(unique_categories)), dtype = np.int64)
lookuptable = np.column_stack([unique_categories, indices])
lookuptable




# #step 3: apply lookuptable to data
# main_result = []
# for i in range(len(Y)):
# #  res = []
# #  for ii in range(len(Y[i])):
#   main_result.append(int(lookuptable[lookuptable[:,0] == Y[i],1][0]))
# #  main_result.append(res)

# #step 4
# y_final = np.array([list(np.zeros(len(unique_categories))) for i in range(len(Y))])

# for i in range(len(Y)):
# #  for ii in range(len(main_result[i])):
#   y_final[i, main_result[i]] = 1

# y_final


X_master_lookup = [] #needs to be outside of any loop reading in data

X_cat = str(X).split(" ")
X_unique = np.unique(np.array(X_cat))
X_master_lookup.extend([item for item in X_unique if item not in X_master_lookup])

X_indices = np.where(np.in1d(X_master_lookup, X_unique))

X_final = np.zeros(len(X_master_lookup))
X_final[X_indices] = 1

X_final_final = X_final.reshape(1,len(X_final))


# X_indices = np.array(range(len(X_unique)), dtype = np.int64)
# X_lookup = np.column_stack([X_unique, X_indices])



# X_main_result = []
# for i in range(len(X_cat)):
#     X_main_result.append(int(X_lookup[X_lookup[:,0] == X_cat[i],1][0]))
# X_final = np.array([list(np.zeros(len(X_unique))) for i in range(len(X_cat))])

# for i in range(len(X_cat)):
#     X_final[i, X_main_result[i]] = 1

# build architecture
inputs = tf.keras.layers.Input(shape=(X_final_final.shape[1],), name='input')
hidden1 = tf.keras.layers.Dense(units=2, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=2, activation="sigmoid", name= 'hidden2')(hidden1)
outputs = tf.keras.layers.Dense(units=y_final_final.shape[1], activation = "sigmoid", name= 'output')(hidden2)

# create model 
model = tf.keras.Model(inputs = inputs, outputs = outputs)

# compile model
model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.SGD(lr = 0.001))


# fit model
model.fit(x = X_final_final, y = y_final_final, batch_size = 1, epochs = 10)

# making a prediction
yhat = model.predict(x = X_final_final)
model.evaluate(x = X_final_final, y = y_final_final)
