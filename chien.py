import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

# Đọc dữ liệu
hf_laydo_df = pd.read_csv("hd_laydo.txt")
hd_dua_df = pd.read_csv("hd_dua.txt")
hd_bamnut_df = pd.read_csv("hd_bamnut.txt")
hd_dat_df = pd.read_csv("hd_dat.txt")

X = []
y = []
no_of_timesteps = 10
#laydo 1000
dataset = hf_laydo_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([1,0,0,0])
#dua 0100
dataset = hd_dua_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,1,0,0])
#dat 0010
dataset = hd_dat_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,1,0])
#bamnut 0001
dataset = hd_bamnut_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,1])




X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 4, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

model.fit(X_train, y_train, epochs=30, batch_size=32,validation_data=(X_test, y_test))
model.save("model.h5")


