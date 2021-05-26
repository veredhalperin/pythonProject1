import pandas as pd
import numpy as np
from CPC18PsychForestPython.CPC15_BEASTpred import CPC15_BEASTpred
from sklearn.metrics import mean_squared_error
from CPC18PsychForestPython.get_PF_Features import lot_shape_convert
from datetime import datetime
import torch
from torch import distributions
from torch.nn import functional as F

def lot_shape_convert2(lot_shape):
    if (lot_shape == [1, 0, 0, 0]).all(): return '-'
    if (lot_shape == [0, 1, 0, 0]).all(): return 'Symm'
    if (lot_shape == [0, 0, 1, 0]).all(): return 'L-skew'
    if (lot_shape == [0, 0, 0, 1]).all(): return 'R-skew'

def create_syntetic_dataset(Data,p=None,s=None,is_sample=False,is_probs=False,original=True):
    nProblems = Data.shape[0]
    Data.index=range(nProblems)
    df = pd.DataFrame()
    for prob in range(0, nProblems, 5):
        if prob > 0:
            print(prob / 5, datetime.now() - start)
        start = datetime.now()
        # read problem's parameters
        Ha = Data['Ha'][prob]
        pHa = Data['pHa'][prob]
        La = Data['La'][prob]
        LotShapeA = lot_shape_convert2(
            Data[['lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A']].values[prob])
        LotNumA = int(Data['LotNumA'][prob])
        Hb = Data['Hb'][prob]
        pHb = Data['pHb'][prob]
        Lb = Data['Lb'][prob]
        LotShapeB = lot_shape_convert2(
            Data[['lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B']].values[prob])
        LotNumB = int(Data['LotNumB'][prob])
        Amb = Data['Amb'][prob]
        Corr = Data['Corr'][prob]
        if is_probs:
            p1=np.array(Data[['Punb','Puni','Ppes','Psig']].values[prob])
            p2 = np.array(Data[['Punb', 'Puni', 'Ppes', 'Psig']].values[prob+1])
            p3 = np.array(Data[['Punb', 'Puni', 'Ppes', 'Psig']].values[prob+2])
            p4 = np.array(Data[['Punb', 'Puni', 'Ppes', 'Psig']].values[prob+3])
            p5 = np.array(Data[['Punb', 'Puni', 'Ppes', 'Psig']].values[prob+4])
        elif is_sample:
            p1=F.softmax(distributions.multivariate_normal.MultivariateNormal(p[prob],torch.matmul(s[prob],s[prob])).rsample()).detach().numpy()
            p2 = F.softmax(distributions.multivariate_normal.MultivariateNormal(p[prob+1],torch.matmul(s[prob+1],s[prob+1])).rsample()).detach().numpy()
            p3 = F.softmax(distributions.multivariate_normal.MultivariateNormal(p[prob+2],torch.matmul(s[prob+2],s[prob+2])).rsample()).detach().numpy()
            p4 = F.softmax(distributions.multivariate_normal.MultivariateNormal(p[prob+3],torch.matmul(s[prob+3],s[prob+3])).rsample()).detach().numpy()
            p5 = F.softmax(distributions.multivariate_normal.MultivariateNormal(p[prob+4],torch.matmul(s[prob+4],s[prob+4])).rsample()).detach().numpy()
        else:
            p1=F.softmax(p[prob]).detach().numpy()
            p2 = F.softmax(p[prob+1]).detach().numpy()
            p3 = F.softmax(p[prob+2]).detach().numpy()
            p4 = F.softmax(p[prob+3]).detach().numpy()
            p5 = F.softmax(p[prob+4]).detach().numpy()
        p1=[0.60924608,0.10653019,0.15455181,0.12967192]
        p2=[0.6933441,0.13316578,0.04766686,0.12582326]
        p3=[0.75092671,0.13190795,0.03360785,0.08355749]
        p4=[0.78345542,0.10062597,0.04773525,0.06818336]
        p5=[0.79453475,0.08921872,0.04986156,0.06638497]

        # convert lot shape: '-'/'Symm'/'L-skew'/'R-skew' to 4 different features for the RF model
        lot_shape_listA = lot_shape_convert(LotShapeA)
        lot_shape_listB = lot_shape_convert(LotShapeB)

        # create features data frame
        feats_labels = (
        'Ha', 'pHa', 'La', 'lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A', 'LotNumA',
        'Hb', 'pHb', 'Lb', 'lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B', 'LotNumB',
        'Amb', 'Corr')
        data_lists = [[Ha, pHa, La], lot_shape_listA, [LotNumA, Hb, pHb, Lb], lot_shape_listB, [LotNumB, Amb, Corr]]
        features_data = [item for sublist in data_lists for item in sublist]
        tmpFeats = pd.DataFrame(features_data, index=feats_labels).T

        # duplicate features data frame as per number of blocks
        Feats = pd.concat([tmpFeats] * 5)

        # get BEAST model prediction as feature
        beastPs = CPC15_BEASTpred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr,p1,p2,p3,p4,p5,original)
        Feats['BEASTpred'] = beastPs

        Feats['block'] = np.arange(1, 6)
        Feats['Feedback'] = 1
        Feats.loc[Feats['block'] == 1, 'Feedback'] = 0

        Feats['GameID'] = [Data['GameID'][prob + i] for i in range(5)]
        Feats['B_rate'] = [Data['B_rate'][prob + i] for i in range(5)]
        df = df.append(Feats)
    return df


if __name__ == '__main__':
    #train = create_syntetic_dataset(pd.read_csv('train_with_probs.csv'), is_probs=True)
    #print("train", mean_squared_error(train['B_rate'], train['BEASTpred']))
    df=pd.read_csv('test_probs.csv')
    #test = create_syntetic_dataset(df, is_probs=True,original=False)
    #print("original", mean_squared_error(test['B_rate'], test['BEASTpred']))
    avg=0
    for i in range(5):
        test = create_syntetic_dataset(df[df['GameID']], is_probs=True,original=False)
        MSE=mean_squared_error(test['B_rate'], test['BEASTpred'])
        print(i, MSE)
        avg+=MSE
    print("avg",avg/10)