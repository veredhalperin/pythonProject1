import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from create_syntetic_dataset import create_syntetic_dataset
from sklearn.metrics import mean_squared_error
from functions import find_weights

torch.autograd.set_detect_anomaly(True)


class attent(torch.nn.Module):
    def __init__(self):
        super(attent, self).__init__()
        self.DropOut = torch.nn.Dropout(p=0.15)
        self.SoftMax = torch.nn.Softmax()
        self.W1 = torch.nn.Linear(120, 60)
        self.W2 = torch.nn.Linear(60, 30)
        self.W3 = torch.nn.Linear(30, 15)
        self.W4 = torch.nn.Linear(15, 8)
        self.W5 = torch.nn.Linear(8, 4)
        """
        self.W1 = torch.nn.Linear(120, 110)
        self.W2 = torch.nn.Linear(110, 95)
        self.W3 = torch.nn.Linear(95, 84)
        """

        

    def forward(self, features,col,
                unb,
                unb_unb,
                unb_unb_unb,
                unb_unb_uni,
                unb_unb_pes,
                unb_unb_sig,
                unb_uni,
                unb_uni_unb,
                unb_uni_uni,
                unb_uni_pes,
                unb_uni_sig,
                unb_pes,
                unb_pes_unb,
                unb_pes_uni,
                unb_pes_pes,
                unb_pes_sig,
                unb_sig,
                unb_sig_unb,
                unb_sig_uni,
                unb_sig_pes,
                unb_sig_sig,
                uni,
                uni_unb,
                uni_unb_unb,
                uni_unb_uni,
                uni_unb_pes,
                uni_unb_sig,
                uni_uni,
                uni_uni_unb,
                uni_uni_uni,
                uni_uni_pes,
                uni_uni_sig,
                uni_pes,
                uni_pes_unb,
                uni_pes_uni,
                uni_pes_pes,
                uni_pes_sig,
                uni_sig,
                uni_sig_unb,
                uni_sig_uni,
                uni_sig_pes,
                uni_sig_sig,
                pes,
                pes_unb,
                pes_unb_unb,
                pes_unb_uni,
                pes_unb_pes,
                pes_unb_sig,
                pes_uni,
                pes_uni_unb,
                pes_uni_uni,
                pes_uni_pes,
                pes_uni_sig,
                pes_pes,
                pes_pes_unb,
                pes_pes_uni,
                pes_pes_pes,
                pes_pes_sig,
                pes_sig,
                pes_sig_unb,
                pes_sig_uni,
                pes_sig_pes,
                pes_sig_sig,
                sig,
                sig_unb,
                sig_unb_unb,
                sig_unb_uni,
                sig_unb_pes,
                sig_unb_sig,
                sig_uni,
                sig_uni_unb,
                sig_uni_uni,
                sig_uni_pes,
                sig_uni_sig,
                sig_pes,
                sig_pes_unb,
                sig_pes_uni,
                sig_pes_pes,
                sig_pes_sig,
                sig_sig,
                sig_sig_unb,
                sig_sig_uni,
                sig_sig_pes,
                sig_sig_sig):
        x = torch.nn.functional.relu(self.W1(features))
        x = self.DropOut(x)
        x = torch.nn.functional.relu(self.W2(x))
        x = self.DropOut(x)
        x = torch.nn.functional.relu(self.W3(x))
        x = self.DropOut(x)
        x = torch.nn.functional.relu(self.W4(x))
        x = self.DropOut(x)
        x = torch.nn.functional.relu(self.W5(x))
        x = self.DropOut(x)
        x = self.SoftMax(x)
        #pred=torch.matmul(x,col)
        #"""
        pred = (
                x[:, 0] * unb +
                x[:, 0] * x[:, 0] * unb_unb +
                x[:, 0] * x[:, 0] * x[:, 0] * unb_unb_unb +
                x[:, 0] * x[:, 0] * x[:, 1] * unb_unb_uni +
                x[:, 0] * x[:, 0] * x[:, 2] * unb_unb_pes +
                x[:, 0] * x[:, 0] * x[:, 3] * unb_unb_sig +
                x[:, 0] * x[:, 1] * unb_uni +
                x[:, 0] * x[:, 1] * x[:, 0] * unb_uni_unb +
                x[:, 0] * x[:, 1] * x[:, 1] * unb_uni_uni +
                x[:, 0] * x[:, 1] * x[:, 2] * unb_uni_pes +
                x[:, 0] * x[:, 1] * x[:, 3] * unb_uni_sig +
                x[:, 0] * x[:, 2] * unb_pes +
                x[:, 0] * x[:, 2] * x[:, 0] * unb_pes_unb +
                x[:, 0] * x[:, 2] * x[:, 1] * unb_pes_uni +
                x[:, 0] * x[:, 2] * x[:, 2] * unb_pes_pes +
                x[:, 0] * x[:, 2] * x[:, 3] * unb_pes_sig +
                x[:, 0] * x[:, 3] * unb_sig +
                x[:, 0] * x[:, 3] * x[:, 0] * unb_sig_unb +
                x[:, 0] * x[:, 3] * x[:, 1] * unb_sig_uni +
                x[:, 0] * x[:, 3] * x[:, 2] * unb_sig_pes +
                x[:, 0] * x[:, 3] * x[:, 3] * unb_sig_sig +
                x[:, 1] * uni +
                x[:, 1] * x[:, 0] * uni_unb +
                x[:, 1] * x[:, 0] * x[:, 0] * uni_unb_unb +
                x[:, 1] * x[:, 0] * x[:, 1] * uni_unb_uni +
                x[:, 1] * x[:, 0] * x[:, 2] * uni_unb_pes +
                x[:, 1] * x[:, 0] * x[:, 3] * uni_unb_sig +
                x[:, 1] * x[:, 1] * uni_uni +
                x[:, 1] * x[:, 1] * x[:, 0] * uni_uni_unb +
                x[:, 1] * x[:, 1] * x[:, 1] * uni_uni_uni +
                x[:, 1] * x[:, 1] * x[:, 2] * uni_uni_pes +
                x[:, 1] * x[:, 1] * x[:, 3] * uni_uni_sig +
                x[:, 1] * x[:, 2] * uni_pes +
                x[:, 1] * x[:, 2] * x[:, 0] * uni_pes_unb +
                x[:, 1] * x[:, 2] * x[:, 1] * uni_pes_uni +
                x[:, 1] * x[:, 2] * x[:, 2] * uni_pes_pes +
                x[:, 1] * x[:, 2] * x[:, 3] * uni_pes_sig +
                x[:, 1] * x[:, 3] * uni_sig +
                x[:, 1] * x[:, 3] * x[:, 0] * uni_sig_unb +
                x[:, 1] * x[:, 3] * x[:, 1] * uni_sig_uni +
                x[:, 1] * x[:, 3] * x[:, 2] * uni_sig_pes +
                x[:, 1] * x[:, 3] * x[:, 3] * uni_sig_sig +
                x[:, 2] * pes +
                x[:, 2] * x[:, 0] * pes_unb +
                x[:, 2] * x[:, 0] * x[:, 0] * pes_unb_unb +
                x[:, 2] * x[:, 0] * x[:, 1] * pes_unb_uni +
                x[:, 2] * x[:, 0] * x[:, 2] * pes_unb_pes +
                x[:, 2] * x[:, 0] * x[:, 3] * pes_unb_sig +
                x[:, 2] * x[:, 1] * pes_uni +
                x[:, 2] * x[:, 1] * x[:, 0] * pes_uni_unb +
                x[:, 2] * x[:, 1] * x[:, 1] * pes_uni_uni +
                x[:, 2] * x[:, 1] * x[:, 2] * pes_uni_pes +
                x[:, 2] * x[:, 1] * x[:, 3] * pes_uni_sig +
                x[:, 2] * x[:, 2] * pes_pes +
                x[:, 2] * x[:, 2] * x[:, 0] * pes_pes_unb +
                x[:, 2] * x[:, 2] * x[:, 1] * pes_pes_uni +
                x[:, 2] * x[:, 2] * x[:, 2] * pes_pes_pes +
                x[:, 2] * x[:, 2] * x[:, 3] * pes_pes_sig +
                x[:, 2] * x[:, 3] * pes_sig +
                x[:, 2] * x[:, 3] * x[:, 0] * pes_sig_unb +
                x[:, 2] * x[:, 3] * x[:, 1] * pes_sig_uni +
                x[:, 2] * x[:, 3] * x[:, 2] * pes_sig_pes +
                x[:, 2] * x[:, 3] * x[:, 3] * pes_sig_sig +
                x[:, 3] * sig +
                x[:, 3] * x[:, 0] * sig_unb +
                x[:, 3] * x[:, 0] * x[:, 0] * sig_unb_unb +
                x[:, 3] * x[:, 0] * x[:, 1] * sig_unb_uni +
                x[:, 3] * x[:, 0] * x[:, 2] * sig_unb_pes +
                x[:, 3] * x[:, 0] * x[:, 3] * sig_unb_sig +
                x[:, 3] * x[:, 1] * sig_uni +
                x[:, 3] * x[:, 1] * x[:, 0] * sig_uni_unb +
                x[:, 3] * x[:, 1] * x[:, 1] * sig_uni_uni +
                x[:, 3] * x[:, 1] * x[:, 2] * sig_uni_pes +
                x[:, 3] * x[:, 1] * x[:, 3] * sig_uni_sig +
                x[:, 3] * x[:, 2] * sig_pes +
                x[:, 3] * x[:, 2] * x[:, 0] * sig_pes_unb +
                x[:, 3] * x[:, 2] * x[:, 1] * sig_pes_uni +
                x[:, 3] * x[:, 2] * x[:, 2] * sig_pes_pes +
                x[:, 3] * x[:, 2] * x[:, 3] * sig_pes_sig +
                x[:, 3] * x[:, 3] * sig_sig +
                x[:, 3] * x[:, 3] * x[:, 0] * sig_sig_unb +
                x[:, 3] * x[:, 3] * x[:, 1] * sig_sig_uni +
                x[:, 3] * x[:, 3] * x[:, 2] * sig_sig_pes +
                x[:, 3] * x[:, 3] * x[:, 3] * sig_sig_sig)
        #"""
        return pred, x


if __name__ == '__main__':
    train = pd.read_csv('SyntheticDataWeights5000.csv')
    real = pd.read_csv('TrainDataWeights210.csv')
    test = pd.read_csv('TestDataWeights60.csv')
    # train = find_weights(pd.read_csv('CPC18PsychForestPython/TrainData210.csv'),1,'TrainData_Weights210.csv')
    # test = find_weights(pd.read_csv('TestData60.csv'),1,'TestData_Weights210.csv')
    col=['Ha','pHa','La','lot_shape__A','lot_shape_symm_A','lot_shape_L_A','lot_shape_R_A','LotNumA','Hb','pHb','Lb','lot_shape__B','lot_shape_symm_B','lot_shape_L_B','lot_shape_R_B','LotNumB','Corr','Amb', 'block', 'Feedback','diffEV', 'diffSDs', 'diffMins', 'diffMaxs', 'diffUV', 'RatioMin', 'SignMax','pBbet_Unbiased1', 'pBbet_UnbiasedFB', 'pBbet_Uniform', 'pBbet_Sign1', 'pBbet_SignFB', 'Dom','diffBEV0', 'diffBEVfb', 'diffSignEV']
    model = attent()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())
    for block in range(1, 6):
        start = datetime.now()
        print("----------------------block: " + str(block) + '----------------------')
        trainb = train[train['block'] == block]
        losses = 0
        print("-----------train--------------")
        for epoch in range(0):
            print("epoch", epoch)
            batch = trainb.sample(n=48)
            B_rate = torch.tensor(np.array(batch['B_rate']), dtype=torch.float32)
            batch = batch.drop(columns=['B_rate', 'GameID'])
            pred, x = model(torch.tensor(batch[batch.columns].values, dtype=torch.float32),
                            torch.transpose(torch.tensor(batch[batch.columns[~batch.columns.isin(col)]].values, dtype=torch.float32), 0, 1),
                            torch.tensor(np.array(batch['unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_unb_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_unb_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_unb_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_unb_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_uni_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_uni_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_uni_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_uni_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_pes_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_pes_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_pes_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_pes_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_sig_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_sig_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_sig_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_sig_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_unb_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_unb_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_unb_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_unb_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_uni_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_uni_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_uni_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_uni_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_pes_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_pes_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_pes_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_pes_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_sig_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_sig_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_sig_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_sig_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_unb_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_unb_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_unb_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_unb_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_uni_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_uni_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_uni_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_uni_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_pes_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_pes_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_pes_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_pes_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_sig_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_sig_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_sig_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_sig_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_unb_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_unb_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_unb_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_unb_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_uni_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_uni_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_uni_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_uni_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_pes_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_pes_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_pes_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_pes_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_sig_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_sig_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_sig_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_sig_sig']), dtype=torch.float32))
            optimizer.zero_grad()
            loss = criterion(pred, B_rate)
                   #+criterion(x[:,0]+x[:,21]+x[:,42]+x[:,63],torch.ones(48,))
            loss.backward()
            optimizer.step()
            # for name, param in model.named_parameters():
            #    print(name, param)
            # print(x)
            losses += loss
            print("loss", losses / (epoch + 1))
        #print(x[:,0],x[:,21],x[:,42],x[:,63])

        realb = real[real['block'] == block]
        print("----------real--------------")
        for epoch in range(1000):
            print("epoch", epoch)
            batch = realb.sample(n=48)
            B_rate = torch.tensor(np.array(batch['B_rate']), dtype=torch.float32)
            batch = batch.drop(columns=['B_rate', 'GameID', 'BEASTpred'])
            pred, x = model(torch.tensor(batch[batch.columns].values, dtype=torch.float32),
                            torch.transpose(torch.tensor(batch[batch.columns[~batch.columns.isin(col)]].values, dtype=torch.float32), 0, 1),
                            torch.tensor(np.array(batch['unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_unb_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_unb_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_unb_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_unb_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_uni_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_uni_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_uni_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_uni_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_pes_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_pes_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_pes_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_pes_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_sig_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_sig_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_sig_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['unb_sig_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_unb_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_unb_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_unb_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_unb_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_uni_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_uni_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_uni_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_uni_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_pes_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_pes_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_pes_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_pes_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_sig_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_sig_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_sig_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['uni_sig_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_unb_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_unb_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_unb_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_unb_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_uni_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_uni_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_uni_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_uni_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_pes_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_pes_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_pes_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_pes_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_sig_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_sig_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_sig_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['pes_sig_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_unb_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_unb_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_unb_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_unb_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_uni_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_uni_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_uni_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_uni_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_pes_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_pes_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_pes_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_pes_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_sig']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_sig_unb']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_sig_uni']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_sig_pes']), dtype=torch.float32),
                            torch.tensor(np.array(batch['sig_sig_sig']), dtype=torch.float32))
            optimizer.zero_grad()
            loss = criterion(pred, B_rate)
                   #+criterion(x[:,0]+x[:,21]+x[:,42]+x[:,63],torch.ones(48,))
            loss.backward()
            optimizer.step()
            # for name, param in model.named_parameters():
            #    print(name, param)
            # print(x)
            losses += loss
            print("loss", losses / (epoch + 1))
        #print(x[:,0],x[:,21],x[:,42],x[:,63])
        trainB_rate = torch.tensor(np.array(realb['B_rate']), dtype=torch.float32)
        batch = realb.drop(columns=['B_rate', 'GameID', 'BEASTpred'])
        pred, x = model(torch.tensor(batch[batch.columns].values, dtype=torch.float32),
                        torch.transpose(torch.tensor(batch[batch.columns[~batch.columns.isin(col)]].values, dtype=torch.float32), 0, 1),
                        torch.tensor(np.array(batch['unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_unb_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_unb_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_unb_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_unb_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_uni_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_uni_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_uni_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_uni_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_pes_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_pes_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_pes_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_pes_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_sig_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_sig_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_sig_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_sig_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_unb_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_unb_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_unb_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_unb_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_uni_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_uni_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_uni_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_uni_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_pes_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_pes_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_pes_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_pes_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_sig_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_sig_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_sig_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_sig_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_unb_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_unb_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_unb_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_unb_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_uni_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_uni_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_uni_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_uni_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_pes_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_pes_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_pes_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_pes_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_sig_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_sig_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_sig_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_sig_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_unb_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_unb_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_unb_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_unb_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_uni_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_uni_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_uni_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_uni_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_pes_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_pes_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_pes_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_pes_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_sig_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_sig_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_sig_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_sig_sig']), dtype=torch.float32))
        #print(x[:,0],x[:,21],x[:,42],x[:,63])
        print("--------------------train loss ", criterion(pred, trainB_rate), "--------------")
        realb['Punb'] = x[:, 0].detach().numpy()
        realb['Puni'] = x[:, 1].detach().numpy()
        realb['Ppes'] = x[:, 2].detach().numpy()
        realb['Psig'] = x[:, 3].detach().numpy()
        if block == 1:
            total_train = realb
        else:
            total_train = total_train.append(realb)
        testb = test[test['block'] == block]
        testB_rate = torch.tensor(np.array(testb['BEASTpred']), dtype=torch.float32)
        batch = testb.drop(columns=['B_rate', 'GameID', 'BEASTpred'])
        pred, x = model(torch.tensor(batch[batch.columns].values, dtype=torch.float32),
                        torch.transpose(torch.tensor(batch[batch.columns[~batch.columns.isin(col)]].values, dtype=torch.float32), 0, 1),
                        torch.tensor(np.array(batch['unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_unb_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_unb_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_unb_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_unb_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_uni_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_uni_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_uni_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_uni_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_pes_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_pes_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_pes_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_pes_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_sig_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_sig_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_sig_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['unb_sig_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_unb_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_unb_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_unb_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_unb_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_uni_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_uni_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_uni_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_uni_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_pes_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_pes_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_pes_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_pes_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_sig_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_sig_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_sig_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['uni_sig_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_unb_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_unb_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_unb_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_unb_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_uni_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_uni_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_uni_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_uni_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_pes_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_pes_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_pes_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_pes_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_sig_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_sig_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_sig_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['pes_sig_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_unb_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_unb_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_unb_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_unb_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_uni_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_uni_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_uni_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_uni_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_pes_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_pes_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_pes_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_pes_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_sig']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_sig_unb']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_sig_uni']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_sig_pes']), dtype=torch.float32),
                        torch.tensor(np.array(batch['sig_sig_sig']), dtype=torch.float32))
        #print(x[:,0], x[:,21], x[:,42], x[:,63])
        print("--------------------test loss ", criterion(pred, testB_rate), "--------------")
        testb['Punb'] = x[:, 0].detach().numpy()
        testb['Puni'] = x[:, 1].detach().numpy()
        testb['Ppes'] = x[:, 2].detach().numpy()
        testb['Psig'] = x[:, 3].detach().numpy()
        if block == 1:
            total_test = testb
        else:
            total_test = total_test.append(testb)
    total_train = total_train.sort_values(by=['GameID', 'block'])
    total_train.to_csv('train_probs.csv', index=False)
    total_test = total_test.sort_values(by=['GameID', 'block'])
    total_test.to_csv('test_probs.csv', index=False)
    total_train = pd.read_csv('train_probs.csv')
    total_test = pd.read_csv('test_probs.csv')
    train = create_syntetic_dataset(total_train,is_probs=True)
    print("----------------train MSE original", mean_squared_error(train['B_rate'], train['BEASTpred']), '---------')
    train = create_syntetic_dataset(total_train,is_probs=True,original=False)
    print("----------------train MSE ", mean_squared_error(train['B_rate'], train['BEASTpred']), '---------')
    train.to_csv('pred_train.csv', index=False)
    test = create_syntetic_dataset(total_test,is_probs=True)
    print("----------------test MSE original", mean_squared_error(test['B_rate'], test['BEASTpred']), '----------')
    test.to_csv('pred_test.csv', index=False)
    test = create_syntetic_dataset(total_test,is_probs=True,original=False)
    print("----------------test MSE ", mean_squared_error(test['B_rate'], test['BEASTpred']), '----------')
    test.to_csv('pred_test.csv', index=False)
