import warnings
import numpy as np

warnings.filterwarnings("ignore")
from _datetime import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"
import logging

logging.basicConfig(filename='/data/home/vhalpern/pythonProject1/TEST_TRAIN2/beast_network_cluster1.log', level=logging.DEBUG)
import pandas as pd

import torch

torch.autograd.set_detect_anomaly(True)





def Sigmoid(x, B):
    return 1 / (1 + torch.exp(-1 * B * x))


def prob_weight(Weights, B, BEV_weights, BEV, ST):
    return torch.dot(Weights, 1 / (1 + torch.exp(-1 * B * (BEV_weights * BEV + ST))))


def calc_probs(probs2):
    probs = torch.tensor([], dtype=torch.float64, requires_grad=True)
    for prob0 in probs2:
        probs = torch.cat((probs, torch.reshape(prob0, (1,))))
        for prob1 in probs2:
            probs = torch.cat((probs, torch.reshape(prob0 * prob1, (1,))))
            for prob2 in probs2:
                probs = torch.cat((probs, torch.reshape(prob0 * prob1 * prob2, (1,))))
    return torch.reshape(probs, (1, probs.shape[0]))


class ProbsNet(torch.nn.Module):
    def __init__(self):
        super(ProbsNet, self).__init__()
        self.probs0 = torch.nn.Parameter(torch.tensor([1.7410,  1.6053, -0.2544,  1.9080], dtype=torch.float64))
        self.probs1 = torch.nn.Parameter(torch.tensor([ 2.6969,  1.5068, -0.0316,  1.5280], dtype=torch.float64))
        self.probs2 = torch.nn.Parameter(torch.tensor([2.8237, 1.4797, 0.0083, 1.4884], dtype=torch.float64))
        self.probs3 = torch.nn.Parameter(torch.tensor([2.9493, 1.4512, 0.0493, 1.4502], dtype=torch.float64))
        self.probs4 = torch.nn.Parameter(torch.tensor([3.0731, 1.4221, 0.0910, 1.4137], dtype=torch.float64))
        self.BEV = torch.nn.Parameter(torch.tensor(0.1516, dtype=torch.float64))
        self.B = torch.nn.Parameter(torch.tensor(0.4130, dtype=torch.float64))

    def forward(self, BEV, ST0, Weight0, ST1, Weight1):
        probs0 = calc_probs(torch.nn.functional.softmax(self.probs0))
        probs1 = calc_probs(torch.nn.functional.softmax(self.probs1))
        probs2 = calc_probs(torch.nn.functional.softmax(self.probs2))
        probs3 = calc_probs(torch.nn.functional.softmax(self.probs3))
        probs4 = calc_probs(torch.nn.functional.softmax(self.probs4))
        tmp0 = torch.tensor([], dtype=torch.float64, requires_grad=True)
        tmp1 = torch.tensor([], dtype=torch.float64, requires_grad=True)
        for i in range(len(Weight0)):
            tmp0 = torch.cat((tmp0, torch.matmul(1 / (1 + torch.exp(-self.B * (torch.nn.functional.relu(self.BEV) * BEV + ST0[i]))), Weight0[i])))
        for i in range(len(Weight1)):
            tmp1 = torch.cat((tmp1, torch.matmul(1 / (1 + torch.exp(-self.B * (torch.nn.functional.relu(self.BEV) * BEV + ST1[i]))), Weight1[i])))
        return torch.mean(torch.cat((
            torch.matmul(probs0, tmp0),
            torch.matmul(probs1, tmp1),
            torch.matmul(probs2, tmp1),
            torch.matmul(probs3, tmp1),
            torch.matmul(probs4, tmp1)
        )))
if __name__ == '__main__':
    model = ProbsNet()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.09)
    Pred = torch.tensor([], dtype=torch.float64, requires_grad=True)
    true = torch.tensor([], dtype=torch.float64, requires_grad=True)
    params = ['probs0', 'probs1', 'probs2', 'probs3', 'probs4', 'BEV', 'B']
    ST1, ST0, Weight1, Weight0 = [], [], [], []
    for k1 in ['unb', 'uni', 'pes', 'sig']:
        Weight1.append('Weight_' + k1)
        ST1.append('ST_' + k1)
        Weight0.append('Weight0_' + k1)
        ST0.append('ST0_' + k1)
        for k2 in ['unb', 'uni', 'pes', 'sig']:
            Weight1.append('Weight_' + k1 + '_' + k2)
            ST1.append('ST_' + k1 + '_' + k2)
            Weight0.append('Weight0_' + k1 + '_' + k2)
            ST0.append('ST0_' + k1 + '_' + k2)
            for k3 in ['unb', 'uni', 'pes', 'sig']:
                Weight1.append('Weight_' + k1 + '_' + k2 + '_' + k3)
                ST1.append('ST_' + k1 + '_' + k2 + '_' + k3)
                Weight0.append('Weight0_' + k1 + '_' + k2 + '_' + k3)
                ST0.append('ST0_' + k1 + '_' + k2 + '_' + k3)
    #"""
    for data in range(99, 9830, 100):
        if data == 99:
            df2 = pd.read_pickle('/data/home/vhalpern/pythonProject1/c13k_selections_ST_' + str(data) + '.pkl')
        else:
            df2 = df2.append(pd.read_pickle('/data/home/vhalpern/pythonProject1/c13k_selections_ST_' + str(data) + '.pkl'))
    df2 = df2.append(pd.read_pickle('/data/home/vhalpern/pythonProject1/c13k_selections_ST_9830.pkl'))
   #"""
    #df2=pd.read_pickle('c13k_selections_ST.pkl')
    df=pd.read_csv('/data/home/vhalpern/pythonProject1/TEST_TRAIN2/c13k_selections_clusters.csv')
    df = df[df['cluster'] == 1]
    df2=df2.merge(df,on='Problem')
    nProblems2=len(df2)
    df2.index=range(nProblems2)
    
    logging.debug(datetime.now())
    for t in range(10):
        df2 = df2.reindex(np.random.permutation(nProblems2))
        print("########## round "+str(t)+" size of data "+str(nProblems2)+"##########")
        logging.debug("########## round "+str(t)+" size of data "+str(nProblems2)+"##########")
        for i in range(nProblems2):
            print(str(i))
            logging.debug(str(i))
            if df2['TEST_TRAIN'][i]=='TRAIN':
                optimizer.zero_grad()

                Pred = torch.cat(
                    (Pred, torch.reshape(model(
                                               df2['BEVb'][i] - df2['BEVa'][i],
                                               [torch.reshape(torch.tensor(df2[j][i],dtype=torch.float64), (1, len(df2[j][i]))) for j in ST0],
                                               [torch.reshape(torch.tensor(df2[j][i],dtype=torch.float64), (len(df2[j][i]),1)) for j in Weight0],
                                               [torch.reshape(torch.tensor(df2[j][i],dtype=torch.float64), (1, len(df2[j][i]))) for j in ST1],
                                               [torch.reshape(torch.tensor(df2[j][i],dtype=torch.float64), (len(df2[j][i]),1)) for j in Weight1]), (1,))))
                true=torch.cat((true,torch.reshape(torch.tensor(df2['B_rate'][i],dtype=torch.float64,requires_grad=True),(1,))))
                if (i+1) % 10 == 0:
                    
                    loss = criterion(Pred, true)
                    
                    
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    j = 0
                    for param in model.parameters():
                        print(params[j] + ' ' + str(param))
                        j+=1
                    Pred = torch.tensor([], dtype=torch.float64, requires_grad=True)
                    true = torch.tensor([], dtype=torch.float64, requires_grad=True)
                    
    with torch.no_grad():
        j = 0
        logging.debug(datetime.now())
        for param in model.parameters():
            logging.debug(params[j] + ' ' + str(param))
            j+=1
        df2['BEASTNET']=df2.apply(lambda x:(model(x['BEVb'] - x['BEVa'],
                                       [torch.reshape(torch.tensor(x[j],dtype=torch.float64), (1, len(x[j]))) for j in ST0],
                                       [torch.reshape(torch.tensor(x[j],dtype=torch.float64), (len(x[j]),1)) for j in Weight0],
                                       [torch.reshape(torch.tensor(x[j],dtype=torch.float64), (1, len(x[j]))) for j in ST1],
                                       [torch.reshape(torch.tensor(x[j],dtype=torch.float64), (len(x[j]),1)) for j in Weight1]).detach().numpy()),axis=1)
        pd.read_csv('/data/home/vhalpern/pythonProject1/TEST_TRAIN2/c13k_selections_clusters.csv').merge(df2[['BEASTNET','Problem']],on='Problem').to_csv('/data/home/vhalpern/pythonProject1/TEST_TRAIN5/c13k_selections_cluster1.csv',index=False)
        
        