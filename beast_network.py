import warnings
warnings.filterwarnings("ignore")
from _datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
logging.basicConfig(filename='beast_network.log', level=logging.DEBUG)
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
torch.autograd.set_detect_anomaly(True)

def Sigmoid(x, B):
    return 1 / (1 + torch.exp(-1 * B * x))


class Not_Dom(torch.nn.Module):
    def __init__(self):
        super(Not_Dom, self).__init__()
        self.BEV = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        self.B = torch.nn.Parameter(torch.tensor(0.29, dtype=torch.float64))
        self.Linear=torch.nn.Parameter(torch.tensor([12.8456, -11.8456], dtype=torch.float64))
        self.Softmax=torch.nn.Softmax()

    def forward(self, BEV, ST1, ST2, ST3, ST4, ST5):
        x= (
                       torch.mean(Sigmoid(self.BEV * BEV + ST1, self.B)) +
                       torch.mean(Sigmoid(self.BEV * BEV + ST2, self.B)) +
                       torch.mean(Sigmoid(self.BEV * BEV + ST3, self.B)) +
                       torch.mean(Sigmoid(self.BEV * BEV + ST4, self.B)) +
                       torch.mean(Sigmoid(self.BEV * BEV + ST5, self.B))
               ) / 5
        return torch.matmul(self.Softmax(self.Linear), torch.cat((torch.reshape(x, (1,)), torch.reshape(
            torch.tensor(0.5, dtype=torch.float64),
            (1,)))))

class Dom(torch.nn.Module):
    def __init__(self):
        super(Dom, self).__init__()
        self.Linear=torch.nn.Parameter(torch.tensor([ 12.8456, -11.8456], dtype=torch.float64))
        self.Softmax=torch.nn.Softmax()

    def forward(self, x):
        return torch.matmul(self.Softmax(self.Linear), torch.cat((torch.reshape(x, (1,)), torch.reshape(
            torch.tensor(0.5, dtype=torch.float64),
            (1,)))))

def prob_weight(Weights, B, BEV_weights, BEV, ST):
    return torch.dot(Weights, 1 / (1 + torch.exp(-1 * B * (BEV_weights * BEV + ST))))

def calc_probs(probs2):
    probs=torch.tensor([],dtype=torch.float64,requires_grad=True)
    for prob0 in probs2:
        probs=torch.cat((probs,torch.reshape(prob0,(1,))))
        for prob1 in probs2:
            probs=torch.cat((probs,torch.reshape(prob0*prob1,(1,))))
            for prob2 in probs2:
                probs=torch.cat((probs,torch.reshape(prob0*prob1*prob2,(1,))))
    return torch.reshape(probs,(1,probs.shape[0]))

class ProbsNet(torch.nn.Module):
    def __init__(self):
        super(ProbsNet, self).__init__()
        self.probs0=torch.nn.Parameter(torch.tensor([2.0327, 1.0198, 1.0240, 0.9935], dtype=torch.float64))
        self.probs1 = torch.nn.Parameter(torch.tensor([2.6661, 1.0168, 1.0200, 0.9971], dtype=torch.float64))
        self.probs2 = torch.nn.Parameter(torch.tensor([2.8173, 1.0160, 1.0189, 0.9978], dtype=torch.float64))
        self.probs3 = torch.nn.Parameter(torch.tensor([2.9183, 1.0154, 1.0181, 0.9981], dtype=torch.float64))
        self.probs4 = torch.nn.Parameter(torch.tensor([2.9689, 1.0151, 1.0177, 0.9983], dtype=torch.float64))
        self.BEV = torch.nn.Parameter(torch.tensor(0.1822, dtype=torch.float64))
        self.B = torch.nn.Parameter(torch.tensor(3.1475, dtype=torch.float64))
        """
        self.probs0=torch.nn.Parameter(torch.tensor([1.9724,  1.7806, -0.5429,  1.8599], dtype=torch.float64))
        self.probs1 = torch.nn.Parameter(torch.tensor([2.7305,  1.5567, -0.1608,  1.5735], dtype=torch.float64))
        self.probs2 = torch.nn.Parameter(torch.tensor([2.8989,  1.5053, -0.0697,  1.5155], dtype=torch.float64))
        self.probs3 = torch.nn.Parameter(torch.tensor([3.0096,  1.4720, -0.0101,  1.4785], dtype=torch.float64))
        self.probs4 = torch.nn.Parameter(torch.tensor([3.0645, 1.4558, 0.0192, 1.4606], dtype=torch.float64))
        self.BEV = torch.nn.Parameter(torch.tensor(0.0404, dtype=torch.float64))
        self.B = torch.nn.Parameter(torch.tensor(0.7004, dtype=torch.float64))
        """

    def forward(self, BEV,ST0,Weight0,ST1,Weight1,Problem):
        print(Problem)
        probs0=calc_probs(torch.nn.functional.softmax(self.probs0))
        probs1=calc_probs(torch.nn.functional.softmax(self.probs1))
        probs2=calc_probs(torch.nn.functional.softmax(self.probs2))
        probs3=calc_probs(torch.nn.functional.softmax(self.probs3))
        probs4=calc_probs(torch.nn.functional.softmax(self.probs4))
        tmp0=torch.tensor([],dtype=torch.float64,requires_grad=True)
        tmp1 = torch.tensor([], dtype=torch.float64, requires_grad=True)
        for i in range(len(Weight0)):
            tmp0=torch.cat((tmp0,torch.matmul(1 / (1 + torch.exp(-self.B * (self.BEV * BEV + ST0[i]))),Weight0[i])))
        for i in range(len(Weight1)):
            tmp1 = torch.cat((tmp1, torch.matmul(1 / (1 + torch.exp(-self.B * (self.BEV * BEV + ST1[i]))), Weight1[i])))
        return torch.mean(torch.cat((
            torch.matmul(probs0,tmp0),
            torch.matmul(probs1,tmp1),
            torch.matmul(probs2,tmp1),
            torch.matmul(probs3,tmp1),
            torch.matmul(probs4,tmp1)
        )))




if __name__ == '__main__':
    for data in range(99, 9831, 100):
        if data == 99:
            df2 = pd.read_pickle('c13k_selections_ST_' + str(data) + '.pkl')
        else:
            df2 = df2.append(pd.read_pickle('c13k_selections_ST_' + str(data) + '.pkl'))
    df2 = df2.append(pd.read_pickle('c13k_selections_ST_9830.pkl'))
    df2.index = range(len(df2))
    df = pd.read_csv('c13k_selections.csv')
    df2['Dom']=df['Dom']
    df2['bRate']=df['bRate']
    df2 = df2[df2['Dom'] != 0]
    df=df[df['Dom']!=0]
    df.index=range(len(df))
    nProblems2=len(df2)
    df2.index=range(nProblems2)
    #model=Dom()
    model=ProbsNet()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    Pred = torch.tensor([], dtype=torch.float64,requires_grad=True)
    true = torch.tensor([], dtype=torch.float64,requires_grad=True)
    params = ['probs0','probs1','probs2','probs3','probs4','BEV', 'B']
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
    """
    for i in range(nProblems2):
        print(str(i))
        logging.debug(str(i))
        Pred = torch.cat(
            (Pred, torch.reshape(model(df2['BEVb'][i] - df2['BEVa'][i],
                                       [torch.reshape(torch.tensor(df2[j][i],dtype=torch.float64), (1, len(df2[j][i]))) for j in ST0],
                                       [torch.reshape(torch.tensor(df2[j][i],dtype=torch.float64), (len(df2[j][i]),1)) for j in Weight0],
                                       [torch.reshape(torch.tensor(df2[j][i],dtype=torch.float64), (1, len(df2[j][i]))) for j in ST1],
                                       [torch.reshape(torch.tensor(df2[j][i],dtype=torch.float64), (len(df2[j][i]),1)) for j in Weight1],
                                       df2['Problem'][i]), (1,))))
        true=torch.cat((true,torch.reshape(torch.tensor(df2['bRate'][i],dtype=torch.float64,requires_grad=True),(1,))))
        if (i+1) % 10 == 0:
            loss = criterion(Pred, true)
            #print("loss "+str(loss))
            #logging.debug("loss "+str(loss))
            loss.backward(retain_graph=True)
            optimizer.step()
            j = 0
            for param in model.parameters():
                #print(params[j] + ' ' + str(param))
                logging.debug(params[j] + ' ' + str(param))
                j += 1
            Pred = torch.tensor([], dtype=torch.float64, requires_grad=True)
            true = torch.tensor([], dtype=torch.float64, requires_grad=True)
    #"""
    with torch.no_grad():
        print(datetime.now())
        df2['Pred_By_Prob_New']=df2.apply(lambda x:(model(x['BEVb'] - x['BEVa'],
                                       [torch.reshape(torch.tensor(x[j],dtype=torch.float64), (1, len(x[j]))) for j in ST0],
                                       [torch.reshape(torch.tensor(x[j],dtype=torch.float64), (len(x[j]),1)) for j in Weight0],
                                       [torch.reshape(torch.tensor(x[j],dtype=torch.float64), (1, len(x[j]))) for j in ST1],
                                       [torch.reshape(torch.tensor(x[j],dtype=torch.float64), (len(x[j]),1)) for j in Weight1],
                                       x['Problem']).detach().numpy()),axis=1)
        df['Pred_By_Prob_New']=df2['Pred_By_Prob_New']
        df.to_csv('c13k_selections_Dom.csv',index=False)
        print(datetime.now())
        print(mean_squared_error(df['bRate'],df['Pred_By_Prob_New']))
    """
    params=['BEV','B','Linear']
    prob=0
    Problem =[]
    for k in range(99, 9900, 100):
        if k==9899:
            k=9830
        print(k)
        logging.debug(str(k))
        df = pd.read_csv(str(k) + '.csv')
        df = df[df['Dom'] == 1]
        nProblems = len(df)
        df.index = range(nProblems)
        for i in range(nProblems):
            optimizer.zero_grad()
            Pred = torch.cat((Pred, torch.reshape(model(df['BEVb'][i] - df['BEVa'][i],
                                                        torch.reshape(torch.tensor(
                                                            [df['STb_' + str(j) + '_0'][i] -
                                                             df['STa_' + str(j) + '_0'][i] for j in range(4000)],
                                                             dtype=torch.float64,requires_grad=True),
                                                            (4000, 1)),
                                                        torch.reshape(torch.tensor(
                                                            [df['STb_' + str(j) + '_1'][i] -
                                                             df['STa_' + str(j) + '_1'][i] for j in range(4000)],
                                                            dtype=torch.float64,requires_grad=True),
                                                            (4000, 1)),
                                                        torch.reshape(torch.tensor(
                                                            [df['STb_' + str(j) + '_2'][i] -
                                                             df['STa_' + str(j) + '_2'][i] for j in range(4000)],
                                                             dtype=torch.float64,requires_grad=True),
                                                            (4000, 1)),
                                                        torch.reshape(torch.tensor(
                                                            [df['STb_' + str(j) + '_3'][i] -
                                                             df['STa_' + str(j) + '_3'][i] for j in range(4000)],
                                                           dtype=torch.float64,requires_grad=True),
                                                            (4000, 1)),
                                                        torch.reshape(torch.tensor(
                                                            [df['STb_' + str(j) + '_4'][i] -
                                                             df['STa_' + str(j) + '_4'][i] for j in range(4000)],
                                                            dtype=torch.float64,requires_grad=True),
                                                            (4000, 1)),
                                                        ), (1,))))
            Problem.append(df['Problem'][i]
            true = torch.cat((true, torch.reshape(
                torch.tensor(df2[df2['Problem'] == df['Problem'][i]]['bRate'][prob],
                             dtype=torch.float64,requires_grad=True), (1,))))
            if prob%2==0:
                loss=criterion(Pred,true)
                loss.backward(retain_graph=True)
                optimizer.step()
                j=0
                for param in model.parameters():
                    print(params[j],param)
                    logging.debug(params[j]+' '+str(param))
                    j+=1
                Pred = torch.tensor([], dtype=torch.float64,requires_grad=True)
                true = torch.tensor([], dtype=torch.float64,requires_grad=True)
            prob+=1
    prob=0
    Pred = torch.tensor([], dtype=torch.float64,requires_grad=True)
    true = torch.tensor([], dtype=torch.float64,requires_grad=True)
    Problem =[]
    with torch.no_grad():
        for k in range(99, 9900, 100):
            if k==9899:
                k=9830
            print(k)
            logging.debug(str(k))
            df = pd.read_csv(str(k) + '.csv')
            df = df[df['Dom'] == 1]
            nProblems = len(df)
            df.index = range(nProblems)
            for i in range(nProblems):
                optimizer.zero_grad()
                Pred = torch.cat((Pred, torch.reshape(model(df['BEVb'][i] - df['BEVa'][i],
                                                            torch.reshape(torch.tensor(
                                                                [df['STb_' + str(j) + '_0'][i] -
                                                                 df['STa_' + str(j) + '_0'][i] for j in range(4000)],
                                                                 dtype=torch.float64,requires_grad=True),
                                                                (4000, 1)),
                                                            torch.reshape(torch.tensor(
                                                                [df['STb_' + str(j) + '_1'][i] -
                                                                 df['STa_' + str(j) + '_1'][i] for j in range(4000)],
                                                                dtype=torch.float64,requires_grad=True),
                                                                (4000, 1)),
                                                            torch.reshape(torch.tensor(
                                                                [df['STb_' + str(j) + '_2'][i] -
                                                                 df['STa_' + str(j) + '_2'][i] for j in range(4000)],
                                                                 dtype=torch.float64,requires_grad=True),
                                                                (4000, 1)),
                                                            torch.reshape(torch.tensor(
                                                                [df['STb_' + str(j) + '_3'][i] -
                                                                 df['STa_' + str(j) + '_3'][i] for j in range(4000)],
                                                               dtype=torch.float64,requires_grad=True),
                                                                (4000, 1)),
                                                            torch.reshape(torch.tensor(
                                                                [df['STb_' + str(j) + '_4'][i] -
                                                                 df['STa_' + str(j) + '_4'][i] for j in range(4000)],
                                                                dtype=torch.float64,requires_grad=True),
                                                                (4000, 1)),
                                                            ), (1,))))
                Problem.append(df['Problem'][i]
                true = torch.cat((true, torch.reshape(
                    torch.tensor(df2[df2['Problem'] == df['Problem'][i]]['bRate'][prob],
                                 dtype=torch.float64,requires_grad=True), (1,))))
                prob+=1
                if prob % 100 == 0 or prob==nProblems2:
                   pd.DataFrame.from_dict({'Problem':Problem,'Pred':Pred.detach().numpy(),'True':true.detach().numpy()}).to_csv('pred2.csv',index=False)
    """