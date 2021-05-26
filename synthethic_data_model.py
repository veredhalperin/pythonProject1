import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Dropout
from random import sample

mse=0
k=10
#,'diffEV', 'diffSDs', 'diffMins', 'diffMaxs', 'diffUV', 'RatioMin', 'SignMax','pBbet_Unbiased1', 'pBbet_UnbiasedFB', 'pBbet_Uniform', 'pBbet_Sign1', 'pBbet_SignFB', 'Dom','diffBEV0', 'diffBEVfb', 'diffSignEV'
test = pd.read_csv('test.csv').drop(['GameID'], axis=1)
y_test = np.array(test['B_rate'])
x_test=np.array(test.drop('B_rate',axis=1))
real = pd.read_csv('TrainData210.csv')
synt = pd.read_csv('train_85000.csv')
synt_games=list(synt['GameID'].unique())
for i in range(k):
 synt_val_games=sample(synt_games,round(len(synt_games)*60/210))
 synt_val=synt[synt['GameID'].isin(synt_val_games)].drop(['GameID'], axis=1)
 synt_train=synt[~synt['GameID'].isin(synt_val_games)].drop(['GameID'], axis=1)
 y_synt_train = np.array(synt_train['B_rate'])
 x_synt_train=np.array(synt_train.drop(['B_rate'],axis=1))
 y_synt_val = np.array(synt_val['B_rate'])
 x_synt_val=np.array(synt_val.drop(['B_rate'],axis=1))
 real_val_games = sample(range(1, 211), 60)
 real_val = real[real['GameID'].isin(real_val_games)].drop(['GameID', 'BEASTpred'], axis=1)
 real_train = real[~real['GameID'].isin(real_val_games)].drop(['GameID', 'BEASTpred'], axis=1)
 y_real_train = np.array(real_train['B_rate'])
 x_real_train = np.array(real_train.drop(['B_rate'], axis=1))
 y_real_val = np.array(real_val['B_rate'])
 x_real_val = np.array(real_val.drop(['B_rate'], axis=1))
 model = Sequential()
 model.add(Dense(200, input_dim=x_synt_train.shape[1], kernel_initializer='normal', activation='relu'))
 model.add(Dropout(rate=0.15))
 model.add(Dense(275, kernel_initializer='normal', activation='relu'))
 model.add(Dropout(rate=0.15))
 model.add(Dense(100, kernel_initializer='normal', activation='relu'))
 model.add(Dropout(rate=0.15))
 model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
 model.compile(loss='mse', optimizer='rmsprop')
 model.fit(x_synt_train,y_synt_train,epochs=1000,batch_size=32,verbose=2,validation_data=(x_synt_val,y_synt_val),callbacks=[EarlyStopping(patience=5),ModelCheckpoint(filepath="model.h5", save_best_only=True)])
 model.fit(x_real_train,y_real_train,epochs=1000,batch_size=32,verbose=2,validation_data=(x_real_val,y_real_val),callbacks=[EarlyStopping(patience=5),ModelCheckpoint(filepath="model.h5", save_best_only=True)])
 model.load_weights("model.h5")
 MSE=mean_squared_error(model.predict(x_test),y_test)
 print(MSE)
 mse+=MSE/k
print('total MSE '+str(mse))