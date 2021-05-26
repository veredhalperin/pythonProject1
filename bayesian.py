import torch
from torch.nn import functional as F
from torch import distributions
import numpy as np
import pandas as pd
from CPC18PsychForestPython.distSample import distSample
from CPC18PsychForestPython.CPC18_getDist import CPC18_getDist
from beast_network import lot_shape_convert2
from datetime import datetime
import math
from sklearn.neighbors import NearestNeighbors
from functions import to_B1
from create_syntetic_dataset import  create_syntetic_dataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer
eps=1e-20
import logging
logging.basicConfig(filename='out.log', level=logging.DEBUG)
class bayes(torch.nn.Module):
    def __init__(self):
        super(bayes, self).__init__()
        """
        self.DropOut = torch.nn.Dropout(p=0.15)
        self.W1 = torch.nn.Linear(35, 17)
        self.W2 = torch.nn.Linear(17, 8)
        self.W3 = torch.nn.Linear(8, 4)
        """
        #self.std = torch.nn.parameter.Parameter(torch.eye(4))
        self.mean = torch.nn.parameter.Parameter(torch.tensor([2,1,1,1],requires_grad=True,dtype=torch.float32))
        self.Sigmoid=torch.nn.Sigmoid()

    def forward(self, outcomeA, outcomeB, BEVa, BEVb, kapa,batch_size,features):
        """
        x = torch.nn.functional.relu(self.W1(features))
        x = self.DropOut(x)
        x = torch.nn.functional.relu(self.W2(x))
        x = self.DropOut(x)
        x = torch.nn.functional.relu(self.W3(x))
        x = self.DropOut(x)
        """
        #print("----------probabilty mean", F.softmax(self.mean), '----------')

        #m = distributions.multivariate_normal.MultivariateNormal(self.mean,torch.matmul(self.std,self.std)).rsample()
        m=self.mean
        tmp=[]
        tmp1=[]
        for b in range(batch_size):
            STa = 0
            STb = 0
            #STa1 = 0
            #STb1 = 0
            for s in range(kapa[b]):
                p = torch.reshape(F.gumbel_softmax(logits=F.log_softmax(m), tau=0.1, hard=True),(4,1))
                p1 = torch.reshape(F.gumbel_softmax(logits=F.log_softmax(torch.tensor([2,1,1,1],requires_grad=True,dtype=torch.float32)), tau=0.1, hard=True), (4, 1))
                STa = STa + torch.matmul(outcomeA[b][:, s], p)
                STb = STb + torch.matmul(outcomeB[b][:, s], p)
                #STa1 = STa1 + torch.matmul(outcomeA[b][:, s], p1)
                #STb1= STb1 + torch.matmul(outcomeB[b][:, s], p1)
            STa = STa / kapa[b]
            STb = STb / kapa[b]
            #STa1 = STa1 / kapa[b]
            #STb1 = STb1 / kapa[b]
            tmp.append(torch.reshape(self.Sigmoid(BEVb[b] - BEVa[b] + STb - STa),(1,)))
            #tmp1.append(torch.reshape(self.Sigmoid(BEVb[b] - BEVa[b] + STb1 - STa1),(1,)))
        return torch.cat(tmp)
def train_model(total,Decision,B_rate,model,optimizer,outcomeA,outcomeB,BEVa,BEVb,kapa,trial,batch_size,sim,block,features_list):
    #print("----------block "+str(block)+' sim '+str(sim)+' trial '+str(trial)+'---------')
    total += 1
    pred=model(outcomeA, outcomeB, BEVa, BEVb, kapa, batch_size, features_list)
    Decision += pred
    pred = Decision/ total
    optimizer.zero_grad()
    #for i in range(batch_size):
        #print('----------pred '+str(pred[i].item())+ ' real '+str(B_rate[i].item())+'----------')
        #print('----------pred1 ' + str(pred1[i].item()) + ' real ' + str(B_rate[i].item()) + '----------')
    loss = criterion(pred, B_rate)
    #print('----------loss '+str(loss.item())+'----------')
    loss.backward(retain_graph=True)
    optimizer.step()
    return total,loss.item(),Decision

df = pd.read_csv('CPC18PsychForestPython/TrainData210.csv')
df2=pd.read_csv('TestData60.csv')
data=df.append(df2)
batch_size = 1
epochs = 100
lr = 0.01
SIGMA = 7
KAPA = 3
BETA = 2.6
GAMA = 0.5
PSI = 0.07
THETA = 1
# run model simulation nSims times
nSims = 4000
nTrials = 25
firstFeedback = 6
nBlocks = 5
model1 = bayes()
model2 = bayes()
model3 = bayes()
model4 = bayes()
model5 = bayes()
optimizer1 = torch.optim.AdamW(model1.parameters(), lr=lr)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=lr)
optimizer3 = torch.optim.AdamW(model3.parameters(), lr=lr)
optimizer4 = torch.optim.AdamW(model4.parameters(), lr=lr)
optimizer5 = torch.optim.AdamW(model5.parameters(), lr=lr)
"""
for i in range(len(dfs)):
    df=dfs[i][dfs[i]['GameID']<211]
    df2=dfs[i][dfs[i]['GameID']>210]
    if len(df)/5>0 and len(df2)/5>0:
        print(names[i],len(df)/5,len(df2)/5)
        logging.info(names[i]+' '+str(len(df)/5)+' '+str(len(df2)/5))
"""
train=Normalizer().fit_transform(np.array(to_B1(df)[['BEASTpred.1','BEASTpred.2','BEASTpred.3','BEASTpred.4','BEASTpred.5']]))
test=Normalizer().fit_transform(np.array(to_B1(df2)[['BEASTpred.1','BEASTpred.2','BEASTpred.3','BEASTpred.4','BEASTpred.5']]))
distances, indices = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train).kneighbors(test)
Data1 = df[df['block'] == 1]
nProblems = Data1.shape[0]
Data1.index = range(nProblems)
Data2 = df[df['block'] == 2]
Data2.index = range(nProblems)
Data3 = df[df['block'] == 3]
Data3.index = range(nProblems)
Data4 = df[df['block'] == 4]
Data4.index = range(nProblems)
Data5 = df[df['block'] == 5]
Data5.index = range(nProblems)
criterion = torch.nn.MSELoss()
columns=Data1.drop(columns=['GameID','Feedback','block','B_rate']).columns
Punb, Puni, Ppes, Psig=[],[],[],[]
probabilties=[]
stds=[]
for prob in range(nProblems):
    logging.info(str(prob))
    #prob = np.random.choice(range(nProblems-batch_size), 1)[0]
    print('----------problems ',[prob+1+i for i in range(batch_size)],'----------')
    start = datetime.now()
    # read problem's parameters
    Ha_list = list(Data1['Ha'][prob:prob + batch_size])
    pHa_list =list(Data1['pHa'][prob:prob + batch_size])
    La_list = list(Data1['La'][prob:prob + batch_size])
    LotShapeA_list = [lot_shape_convert2(x) for x in
        Data1[['lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A']].values[prob:prob + batch_size]]
    LotNumA_list = [int(x) for x in Data1['LotNumA'][prob:prob + batch_size]]
    Hb_list = list(Data1['Hb'][prob:prob + batch_size])
    pHb_list = list(Data1['pHb'][prob:prob + batch_size])
    Lb_list = list(Data1['Lb'][prob:prob + batch_size])
    LotShapeB_list = [lot_shape_convert2(x) for x in
        Data1[['lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B']].values[prob:prob + batch_size]]
    LotNumB_list = [int(x) for x in Data1['LotNumB'][prob:prob + batch_size]]
    Amb_list = list(Data1['Amb'][prob:prob + batch_size])
    Corr_list = list(Data1['Corr'][prob:prob + batch_size])
    B_rate1 = torch.tensor(list(Data1['B_rate'][prob:prob + batch_size]), dtype=torch.float32)
    B_rate2 = torch.tensor(list(Data2['B_rate'][prob:prob + batch_size]),dtype=torch.float32)
    B_rate3 = torch.tensor(list(Data3['B_rate'][prob:prob + batch_size]),dtype=torch.float32)
    B_rate4 = torch.tensor(list(Data4['B_rate'][prob:prob + batch_size]),dtype=torch.float32)
    B_rate5 = torch.tensor(list(Data5['B_rate'][prob:prob + batch_size]),dtype=torch.float32)
    features_list=torch.tensor(torch.tensor([x for x in Data1[columns].values[prob:prob + batch_size]],dtype=torch.float32),dtype=torch.float32)
    DistA_list = []
    DistB_list = []
    nA_list = []
    nB_list = []
    ambiguous_list = []
    BEVa_list = []
    BEVb_list = []
    pEstB_list = []
    Range_list = []
    SignMax_list = []
    RatioMin_list = []
    MinA_list = []
    MinB_list = []
    Decision1 = 0
    total1 = 0
    Decision2 = 0
    total2 = 0
    Decision3 = 0
    total3 = 0
    Decision4 = 0
    total4 = 0
    Decision5 = 0
    total5 = 0
    for game in range(batch_size):
        Ha = Ha_list[game]
        pHa = pHa_list[game]
        La = La_list[game]
        LotShapeA = LotShapeA_list[game]
        LotNumA = LotNumA_list[game]
        Hb = Hb_list[game]
        pHb = pHb_list[game]
        Lb = Lb_list[game]
        LotShapeB = LotShapeB_list[game]
        LotNumB = LotNumB_list[game]
        Amb=Amb_list[game]
        Corr=Corr_list[game]

        # get both options' distributions
        DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
        DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)

        # Useful variables
        nA = DistA.shape[0]  # num outcomes in A
        nB = DistB.shape[0]  # num outcomes in B

        if Amb == 1:
            ambiguous = True
        else:
            ambiguous = False

        MinA = DistA[0, 0]
        MinB = DistB[0, 0]
        MaxOutcome = np.maximum(DistA[nA - 1, 0], DistB[nB - 1, 0])
        SignMax = np.sign(MaxOutcome)

        if MinA == MinB:
            RatioMin = 1
        elif np.sign(MinA) == np.sign(MinB):
            RatioMin = min(abs(MinA), abs(MinB)) / max(abs(MinA), abs(MinB))
        else:
            RatioMin = 0

        Range = MaxOutcome - min(MinA, MinB)

        BEVa = np.matrix.dot(DistA[:, 0], DistA[:, 1])
        pEstB = DistB[:, 1]
        BEVb = np.matrix.dot(DistB[:, 0], pEstB)

        DistA_list.append(DistA)
        DistB_list.append(DistB)
        nA_list.append(nA)
        nB_list.append(nB)
        ambiguous_list.append(ambiguous)
        MinA_list.append(MinA)
        MinB_list.append(MinB)
        pEstB_list.append(pEstB)
        SignMax_list.append(SignMax)
        RatioMin_list.append(RatioMin)
        Range_list.append(Range)
        BEVa_list.append([BEVa for i in range(nTrials)])
        BEVb_list.append([BEVb for i in range(nTrials)])
    loss1 = 1
    loss2 = 1
    loss3 = 1
    loss4 = 1
    loss5 = 1
    for sim in range(12):
        if loss1>0.0077 or loss2>0.0077 or loss3>0.0077 or loss4>0.0077 or loss5>0.0077:
            kapa_list = []
            outcomeA_list = []
            outcomeB_list = []
            for game in range(batch_size):
                Amb = Amb_list[game]
                Corr = Corr_list[game]
                DistA = DistA_list[game]
                DistB = DistB_list[game]
                nA = nA_list[game]
                nB = nB_list[game]
                ambiguous = ambiguous_list[game]
                MinA = MinA_list[game]
                MinB = MinB_list[game]
                SignMax = SignMax_list[game]
                RatioMin = RatioMin_list[game]
                Range = Range_list[game]
                BEVa = BEVa_list[game][0]
                BEVb = BEVb_list[game][0]
                pEstB = pEstB_list[game]

                # draw personal traits
                sigma = SIGMA * np.random.uniform(size=1)
                kapa = np.random.choice(range(1, KAPA + 1), 1)
                beta = BETA * np.random.uniform(size=1)
                gama = GAMA * np.random.uniform(size=1)
                psi = PSI * np.random.uniform(size=1)
                theta = THETA * np.random.uniform(size=1)

                nfeed = 0  # "t"; number of outcomes with feedback so far

                ObsPay = np.zeros(shape=(nTrials - firstFeedback + 1, 2))  # observed outcomes in A (col1) and B (col2)

                if ambiguous:
                    UEVb = np.matrix.dot(DistB[:, 0], np.repeat([1 / nB], nB))
                    BEVb = (1 - psi) * (UEVb + BEVa) / 2 + psi * MinB
                    for i in range(5):
                        BEVb_list[game][i] = BEVb[0]
                    pEstB = np.repeat([float(nB)], 1)  # estimation of probabilties in Amb
                    t_SPminb = (BEVb - np.mean(DistB[1:nB + 1, 0])) / (MinB - np.mean(DistB[1:nB + 1, 0]))
                    if t_SPminb < 0:
                        pEstB[0] = 0
                    elif t_SPminb > 1:
                        pEstB[0] = 1
                    else:
                        pEstB[0] = t_SPminb

                    # Add nb-1 rows to pEstB:
                    pEstB = np.append(pEstB, np.repeat((1 - pEstB[0]) / (nB - 1), nB - 1))

                outcomeAs = []
                outcomeBs = []
                for trial in range(nTrials):
                    if trial >= firstFeedback - 1:
                        #  got feedback
                        nfeed += 1

                        rndNumObs = np.random.uniform(size=1)
                        ObsPay[nfeed - 1, 0] = distSample(DistA[:, 0], DistA[:, 1], rndNumObs)  # draw outcome from A
                        if Corr == 1:
                            ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], rndNumObs)
                        elif Corr == -1:
                            ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], 1 - rndNumObs)
                        else:
                            # draw outcome from B
                            ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], np.random.uniform(size=1))
                        if ambiguous:
                            BEVb = (1 - 1 / (nTrials - firstFeedback + 1)) * BEVb + 1 / (nTrials - firstFeedback + 1) * \
                                   ObsPay[nfeed - 1, 1]
                            BEVb_list[game][trial] = BEVb[0]

                    outcomeA = np.zeros(shape=(4, kapa[0]))
                    outcomeB = np.zeros(shape=(4, kapa[0]))
                    for s in range(kapa[0]):
                        rndNum = np.random.uniform(size=2)
                        if nfeed == 0:
                            outcomeA[0][s] += distSample(DistA[:, 0], DistA[:, 1], rndNum[1])
                            outcomeB[0][s] += distSample(DistB[:, 0], pEstB, rndNum[1])
                        else:
                            uniprobs = np.repeat([1 / nfeed], nfeed)
                            outcomeA[0][s] += distSample(ObsPay[0:nfeed, 0], uniprobs, rndNum[1])
                            outcomeB[0][s] += distSample(ObsPay[0:nfeed, 1], uniprobs, rndNum[1])
                        outcomeA[1][s] += distSample(DistA[:, 0], np.repeat([1 / nA], nA), rndNum[1])
                        outcomeB[1][s] += distSample(DistB[:, 0], np.repeat([1 / nB], nB), rndNum[1])
                        if SignMax > 0 and RatioMin < gama:
                            outcomeA[2][s] += MinA
                            outcomeB[2][s] += MinB
                        else:
                            outcomeA[2][s] += distSample(DistA[:, 0], np.repeat([1 / nA], nA), rndNum[1])
                            outcomeB[2][s] += distSample(DistB[:, 0], np.repeat([1 / nB], nB), rndNum[1])
                        if nfeed == 0:
                            outcomeA[3][s] += Range * distSample(np.sign(DistA[:, 0]), DistA[:, 1], rndNum[1])
                            outcomeB[3][s] += Range * distSample(np.sign(DistB[:, 0]), pEstB, rndNum[1])
                        else:
                            uniprobs = np.repeat([1 / nfeed], nfeed)
                            outcomeA[3][s] += Range * distSample(np.sign(ObsPay[0:nfeed, 0]), uniprobs, rndNum[1])
                            outcomeB[3][s] += Range * distSample(np.sign(ObsPay[0:nfeed, 1]), uniprobs, rndNum[1])
                    outcomeAs.append(outcomeA)
                    outcomeBs.append(outcomeB)
                kapa_list.append(kapa[0])
                outcomeA_list.append(outcomeAs)
                outcomeB_list.append(outcomeBs)
            for trial in range(nTrials):
                outcomeA=[torch.tensor(x[trial], requires_grad=True, dtype=torch.float32) for x in outcomeA_list]
                outcomeB = [torch.tensor(x[trial], requires_grad=True, dtype=torch.float32) for x in outcomeB_list]
                BEVa=[x[trial] for x in BEVa_list]
                BEVb=[x[trial] for x in BEVb_list]
                if trial < 5:
                    if loss1 > 0.0077:
                        total1,loss1,Decision1=train_model(total1,Decision1,B_rate1,model1,optimizer1,outcomeA,outcomeB,BEVa,BEVb,kapa_list,trial,batch_size,sim,1,features_list)
                elif trial < 10:
                    if loss2 > 0.0077:
                        total2,loss2,Decision2=train_model(total2,Decision2,B_rate2,model2,optimizer2,outcomeA,outcomeB,BEVa,BEVb,kapa_list,trial,batch_size,sim,2,features_list)
                elif trial < 15:
                    if loss3 > 0.0077:
                        total3,loss3,Decision3=train_model(total3,Decision3,B_rate3,model3,optimizer3,outcomeA,outcomeB,BEVa,BEVb,kapa_list,trial,batch_size,sim,3,features_list)
                elif trial < 20:
                    if loss4 > 0.0077:
                        total4,loss4,Decision4=train_model(total4,Decision4,B_rate4,model4,optimizer4,outcomeA,outcomeB,BEVa,BEVb,kapa_list,trial,batch_size,sim,4,features_list)
                elif trial < 25:
                    if loss5 > 0.0077:
                        total5,loss5,Decision5=train_model(total5,Decision5,B_rate5,model5,optimizer5,outcomeA,outcomeB,BEVa,BEVb,kapa_list,trial,batch_size,sim,5,features_list)

    models=[model1,model2,model3,model4,model5]
    for i in range(5):
        for name, param in models[i].named_parameters():
            if name=='mean':
                probabilties.append(param)
            else:
                stds.append(param)
test_probabilitiess=[]
test_stds=[]
for i in indices:
    for j in range(5):
        test_probabilitiess.append(probabilties[5*i[0]+j])
#test_stds.append(stds[j][i])
train = create_syntetic_dataset(df,p=probabilties,original=False)
test = create_syntetic_dataset(df2,p=test_probabilitiess,original=False)
print("train", mean_squared_error(train['B_rate'], train['BEASTpred']))
print("test", mean_squared_error(test['B_rate'], test['BEASTpred']))
test.to_csv('learned_prob_test.csv',index=False)
train.to_csv('learned_prob_train.csv',index=False)
non_amb_train=train[train['Amb']==0]
amb_train=train[train['Amb']==1]
non_amb_test=test[test['Amb']==0]
amb_test=test[test['Amb']==1]
print("non amb")
print("train", mean_squared_error(non_amb_train['B_rate'], non_amb_train['BEASTpred']))
print("test", mean_squared_error(non_amb_test['B_rate'], non_amb_test['BEASTpred']))
print("amb")
print("train", mean_squared_error(amb_train['B_rate'], amb_train['BEASTpred']))
print("test", mean_squared_error(amb_test['B_rate'], amb_test['BEASTpred']))
logging.info("test "+str(mean_squared_error(test['B_rate'], test['BEASTpred'])))
test = create_syntetic_dataset(df,p=probabilties)
test = create_syntetic_dataset(df2,p=test_probabilitiess)
print("train original", mean_squared_error(train['B_rate'], train['BEASTpred']))
print("test original", mean_squared_error(test['B_rate'], test['BEASTpred']))
non_amb_train=train[train['Amb']==0]
amb_train=train[train['Amb']==1]
non_amb_test=test[test['Amb']==0]
amb_test=test[test['Amb']==1]
print("non amb")
print("train original", mean_squared_error(non_amb_train['B_rate'], non_amb_train['BEASTpred']))
print("test original", mean_squared_error(non_amb_test['B_rate'], non_amb_test['BEASTpred']))
print("amb")
print("train original", mean_squared_error(amb_train['B_rate'], amb_train['BEASTpred']))
print("test original", mean_squared_error(amb_test['B_rate'], amb_test['BEASTpred']))
logging.info("test original "+str(mean_squared_error(test['B_rate'], test['BEASTpred'])))


"""
print("----------test----------")
Data1 = df2[df2['block'] == 1]
nProblems = Data1.shape[0]
Data1.index = range(nProblems)
Data2 = df2[df2['block'] == 2]
Data2.index = range(nProblems)
Data3 = df2[df2['block'] == 3]
Data3.index = range(nProblems)
Data4 = df2[df2['block'] == 4]
Data4.index = range(nProblems)
Data5 = df2[df2['block'] == 5]
tmp=[]
for prob in range(nProblems):
    print(prob)
    Ha_list = list(Data1['Ha'][prob:prob + batch_size])
    pHa_list = list(Data1['pHa'][prob:prob + batch_size])
    La_list = list(Data1['La'][prob:prob + batch_size])
    LotShapeA_list = [lot_shape_convert2(x) for x in
                      Data1[['lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A']].values[
                      prob:prob + batch_size]]
    LotNumA_list = [int(x) for x in Data1['LotNumA'][prob:prob + batch_size]]
    Hb_list = list(Data1['Hb'][prob:prob + batch_size])
    pHb_list = list(Data1['pHb'][prob:prob + batch_size])
    Lb_list = list(Data1['Lb'][prob:prob + batch_size])
    LotShapeB_list = [lot_shape_convert2(x) for x in
                      Data1[['lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B']].values[
                      prob:prob + batch_size]]
    LotNumB_list = [int(x) for x in Data1['LotNumB'][prob:prob + batch_size]]
    Amb_list = list(Data1['Amb'][prob:prob + batch_size])
    Corr_list = list(Data1['Corr'][prob:prob + batch_size])
    B_rate1 = torch.tensor(list(Data1['B_rate'][prob:prob + batch_size]), dtype=torch.float32)
    B_rate2 = torch.tensor(list(Data2['B_rate'][prob:prob + batch_size]), dtype=torch.float32)
    B_rate3 = torch.tensor(list(Data3['B_rate'][prob:prob + batch_size]), dtype=torch.float32)
    B_rate4 = torch.tensor(list(Data4['B_rate'][prob:prob + batch_size]), dtype=torch.float32)
    B_rate5 = torch.tensor(list(Data5['B_rate'][prob:prob + batch_size]), dtype=torch.float32)
    features_list = torch.tensor(
        torch.tensor([x for x in Data1[columns].values[prob:prob + batch_size]], dtype=torch.float32),
        dtype=torch.float32)
    DistA_list = []
    DistB_list = []
    nA_list = []
    nB_list = []
    ambiguous_list = []
    BEVa_list = []
    BEVb_list = []
    pEstB_list = []
    Range_list = []
    SignMax_list = []
    RatioMin_list = []
    MinA_list = []
    MinB_list = []
    Decision1 = 0
    Decision2 = 0
    Decision3 = 0
    Decision4 = 0
    Decision5 = 0
    for game in range(batch_size):
        Ha = Ha_list[game]
        pHa = pHa_list[game]
        La = La_list[game]
        LotShapeA = LotShapeA_list[game]
        LotNumA = LotNumA_list[game]
        Hb = Hb_list[game]
        pHb = pHb_list[game]
        Lb = Lb_list[game]
        LotShapeB = LotShapeB_list[game]
        LotNumB = LotNumB_list[game]
        Amb=Amb_list[game]
        Corr=Corr_list[game]

    # get both options' distributions
    DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
    DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)

    # Useful variables
    nA = DistA.shape[0]  # num outcomes in A
    nB = DistB.shape[0]  # num outcomes in B

    if Amb == 1:
        ambiguous = True
    else:
        ambiguous = False

    MinA = DistA[0, 0]
    MinB = DistB[0, 0]
    MaxOutcome = np.maximum(DistA[nA - 1, 0], DistB[nB - 1, 0])
    SignMax = np.sign(MaxOutcome)

    if MinA == MinB:
        RatioMin = 1
    elif np.sign(MinA) == np.sign(MinB):
        RatioMin = min(abs(MinA), abs(MinB)) / max(abs(MinA), abs(MinB))
    else:
        RatioMin = 0

    Range = MaxOutcome - min(MinA, MinB)

    BEVa = np.matrix.dot(DistA[:, 0], DistA[:, 1])
    pEstB = DistB[:, 1]
    BEVb = np.matrix.dot(DistB[:, 0], pEstB)

    DistA_list.append(DistA)
    DistB_list.append(DistB)
    nA_list.append(nA)
    nB_list.append(nB)
    ambiguous_list.append(ambiguous)
    MinA_list.append(MinA)
    MinB_list.append(MinB)
    pEstB_list.append(pEstB)
    SignMax_list.append(SignMax)
    RatioMin_list.append(RatioMin)
    Range_list.append(Range)
    BEVa_list.append([BEVa for i in range(nTrials)])
    BEVb_list.append([BEVb for i in range(nTrials)])
    for sim in range(0, 100):
        print("----------simulation "+str(sim)+'----------')
        kapa_list = []
        outcomeA_list = []
        outcomeB_list = []
        for game in range(batch_size):
            Amb = Amb_list[game]
            Corr = Corr_list[game]
            DistA = DistA_list[game]
            DistB = DistB_list[game]
            nA = nA_list[game]
            nB = nB_list[game]
            ambiguous = ambiguous_list[game]
            MinA = MinA_list[game]
            MinB = MinB_list[game]
            SignMax = SignMax_list[game]
            RatioMin = RatioMin_list[game]
            Range = Range_list[game]
            BEVa = BEVa_list[game][0]
            BEVb = BEVb_list[game][0]
            pEstB = pEstB_list[game]

            # draw personal traits
            sigma = SIGMA * np.random.uniform(size=1)
            kapa = np.random.choice(range(1, KAPA + 1), 1)
            beta = BETA * np.random.uniform(size=1)
            gama = GAMA * np.random.uniform(size=1)
            psi = PSI * np.random.uniform(size=1)
            theta = THETA * np.random.uniform(size=1)

            nfeed = 0  # "t"; number of outcomes with feedback so far

            ObsPay = np.zeros(shape=(nTrials - firstFeedback + 1, 2))  # observed outcomes in A (col1) and B (col2)

            if ambiguous:
                UEVb = np.matrix.dot(DistB[:, 0], np.repeat([1 / nB], nB))
                BEVb = (1 - psi) * (UEVb + BEVa) / 2 + psi * MinB
                for i in range(5):
                    BEVb_list[game][i] = BEVb[0]
                pEstB = np.repeat([float(nB)], 1)  # estimation of probabilties in Amb
                t_SPminb = (BEVb - np.mean(DistB[1:nB + 1, 0])) / (MinB - np.mean(DistB[1:nB + 1, 0]))
                if t_SPminb < 0:
                    pEstB[0] = 0
                elif t_SPminb > 1:
                    pEstB[0] = 1
                else:
                    pEstB[0] = t_SPminb

                # Add nb-1 rows to pEstB:
                pEstB = np.append(pEstB, np.repeat((1 - pEstB[0]) / (nB - 1), nB - 1))

            outcomeAs = []
            outcomeBs = []
            for trial in range(nTrials):
                if trial >= firstFeedback - 1:
                    #  got feedback
                    nfeed += 1

                    rndNumObs = np.random.uniform(size=1)
                    ObsPay[nfeed - 1, 0] = distSample(DistA[:, 0], DistA[:, 1], rndNumObs)  # draw outcome from A
                    if Corr == 1:
                        ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], rndNumObs)
                    elif Corr == -1:
                        ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], 1 - rndNumObs)
                    else:
                        # draw outcome from B
                        ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], np.random.uniform(size=1))
                    if ambiguous:
                        BEVb = (1 - 1 / (nTrials - firstFeedback + 1)) * BEVb + 1 / (nTrials - firstFeedback + 1) * \
                               ObsPay[nfeed - 1, 1]
                        BEVb_list[game][trial] = BEVb[0]

                outcomeA = np.zeros(shape=(4, kapa[0]))
                outcomeB = np.zeros(shape=(4, kapa[0]))
                for s in range(kapa[0]):
                    rndNum = np.random.uniform(size=2)
                    if nfeed == 0:
                        outcomeA[0][s] += distSample(DistA[:, 0], DistA[:, 1], rndNum[1])
                        outcomeB[0][s] += distSample(DistB[:, 0], pEstB, rndNum[1])
                    else:
                        uniprobs = np.repeat([1 / nfeed], nfeed)
                        outcomeA[0][s] += distSample(ObsPay[0:nfeed, 0], uniprobs, rndNum[1])
                        outcomeB[0][s] += distSample(ObsPay[0:nfeed, 1], uniprobs, rndNum[1])
                    outcomeA[1][s] += distSample(DistA[:, 0], np.repeat([1 / nA], nA), rndNum[1])
                    outcomeB[1][s] += distSample(DistB[:, 0], np.repeat([1 / nB], nB), rndNum[1])
                    if SignMax > 0 and RatioMin < gama:
                        outcomeA[2][s] += MinA
                        outcomeB[2][s] += MinB
                    else:
                        outcomeA[2][s] += distSample(DistA[:, 0], np.repeat([1 / nA], nA), rndNum[1])
                        outcomeB[2][s] += distSample(DistB[:, 0], np.repeat([1 / nB], nB), rndNum[1])
                    if nfeed == 0:
                        outcomeA[3][s] += Range * distSample(np.sign(DistA[:, 0]), DistA[:, 1], rndNum[1])
                        outcomeB[3][s] += Range * distSample(np.sign(DistB[:, 0]), pEstB, rndNum[1])
                    else:
                        uniprobs = np.repeat([1 / nfeed], nfeed)
                        outcomeA[3][s] += Range * distSample(np.sign(ObsPay[0:nfeed, 0]), uniprobs, rndNum[1])
                        outcomeB[3][s] += Range * distSample(np.sign(ObsPay[0:nfeed, 1]), uniprobs, rndNum[1])
                outcomeAs.append(outcomeA)
                outcomeBs.append(outcomeB)
            kapa_list.append(kapa[0])
            outcomeA_list.append(outcomeAs)
            outcomeB_list.append(outcomeBs)
        for trial in range(nTrials):
            outcomeA=[torch.tensor(x[trial], requires_grad=True, dtype=torch.float32) for x in outcomeA_list]
            outcomeB = [torch.tensor(x[trial], requires_grad=True, dtype=torch.float32) for x in outcomeB_list]
            BEVa=[x[trial] for x in BEVa_list]
            BEVb=[x[trial] for x in BEVb_list]
            Decision1+=model1(outcomeA,outcomeB,BEVa,BEVb,kapa,batch_size,features_list)
            Decision2+=model2(outcomeA,outcomeB,BEVa,BEVb,kapa,batch_size,features_list)
            Decision3+=model3(outcomeA,outcomeB,BEVa,BEVb,kapa,batch_size,features_list)
            Decision4+=model4(outcomeA,outcomeB,BEVa,BEVb,kapa,batch_size,features_list)
            Decision5+=model5(outcomeA,outcomeB,BEVa,BEVb,kapa,batch_size,features_list)
tmp.append(Decision1/nTrials*nSims)
tmp.append(Decision2/nTrials*nSims)
tmp.append(Decision3/nTrials*nSims)
tmp.append(Decision4/nTrials*nSims)
tmp.append(Decision5/nTrials*nSims)
print("----------MSE ", mean_squared_error(df2['B_rate'],tmp),"---------")
"""







