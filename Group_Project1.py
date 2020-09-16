"""
BZAN 554 Group Project 1
Tim Hengst, Price McGinnis, Luke McConnell
September 2020
"""

#################################
#### CODE FOR MODEL BUILDING ####
#################################

# IMPORT MODULES
import numpy as np
import gzip
import os
import tensorflow as tf

# SET WORKING DIRECTORY
os.chdir('/Users/lukemcconnell/Downloads')

# DEFINE PARSE FUNCTION
def parse(path):
    """
    This function is used to read in data from the gzip file
    """
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)
        
# READ IN FIRST 100,000 ROWS, CLEAN THEM, AND CONVERY TO DUMMY ARRAY
i=0
df = {}
master_lookup = []

for d in parse(r'/Users/lukemcconnell/Downloads/meta_Clothing_Shoes_and_Jewelry.json.gz'):
    i += 1
    print(i)
    X = np.array(d['title']) 
    Y = np.array(d['category']) 

    y_unique = np.unique(np.array(Y))

    X_cat = str(X).split(" ")
    X_unique = np.unique(np.array(X_cat))

    # Remove all values that are duplicates or have a length longer then 75 characters
    master_lookup.extend([item for item in y_unique if (item not in master_lookup) & (len(item) < 75)])
    master_lookup.extend([item for item in X_unique if (item not in master_lookup) & (len(item) < 75)])

    if i == 100000:
        break

# BUILD ARCHITECTURE
inputs = tf.keras.layers.Input(shape=(len(master_lookup),), name='input')
hidden1 = tf.keras.layers.Dense(units=2, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=2, activation="sigmoid", name= 'hidden2')(hidden1)
outputs = tf.keras.layers.Dense(units=len(master_lookup), activation = "sigmoid", name= 'output')(hidden2)   

# CREATE MODEL 
model = tf.keras.Model(inputs = inputs, outputs = outputs)

# COMPILE MODEL
model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.SGD(lr = 0.001))

# RUN MODEL TO PREDICT ROW BY ROW FOR FIRST 100,000 ROWS
i=0
for d in parse(r'/Users/lukemcconnell/Downloads/meta_Clothing_Shoes_and_Jewelry.json.gz'):
    i += 1
    print(i)
    X = np.array(d['title']) 
    Y = np.array(d['category']) 

    # Overall prepossesing - converts to dummy array
    y_cat_flat = [item for item in Y]

    y_unique = np.unique(np.array(y_cat_flat))

    X_cat = str(X).split(" ")
    X_unique = np.unique(np.array(X_cat))

    y_indices = np.where(np.in1d(master_lookup, y_unique))
    X_indices = np.where(np.in1d(master_lookup, X_unique))

    y_final = np.zeros(len(master_lookup))
    y_final[y_indices] = 1
    X_final = np.zeros(len(master_lookup))
    X_final[X_indices] = 1

    y_final_final = y_final.reshape(1,len(y_final))
    X_final_final = X_final.reshape(1,len(X_final))

    # fit model for each record
    model.fit(x = X_final_final, y = y_final_final, batch_size = 1, epochs = 10)
    if i == 100000:
        break
        
###############################
#### CODE FOR PRESENTATION ####
###############################

# MAKING A PREDICTION
yhat = model.predict(x = X_final_final)
yhat_copy = yhat.copy()

res_indices = yhat_copy.argsort()[0][-5:]
for ind in res_indices:
    print(master_lookup[ind])

# EVALUATING MODEL PERFORMANCE
model.evaluate(x = X_final_final, y = y_final_final)
