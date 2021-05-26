from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from datetime import datetime
import logging
logging.basicConfig(filename='out.log', level=logging.DEBUG)
from CPC18PsychForestPython.distSample import distSample
from CPC18PsychForestPython.CPC18_getDist import  CPC18_getDist
import math
import numpy as np
from CPC18PsychForestPython.CPC15_isStochasticDom import CPC15_isStochasticDom
from CPC18PsychForestPython.get_pBetter import get_pBetter
import pandas as pd
from create_syntetic_dataset import create_syntetic_dataset
def lot_shape_convert(lot_shape):
    return {
        '-': [1, 0, 0, 0],
        'Symm': [0, 1, 0, 0],
        'L-skew': [0, 0, 1, 0],
        'R-skew': [0, 0, 0, 1],
    }
def getSD(vals, probs):
    m = np.matrix.dot(vals, probs.T)
    sqds = np.power((vals - m), 2)
    var = np.matrix.dot(probs, sqds.T)
    return math.sqrt(var)
def get_PF_Features(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
    # Finds the values of the engineered features that are part of Psychological Forest
    # Gets as input the parameters defining the choice problem in CPC18 and returns
    # as output a pandas data frame with this problem's features

    # Compute "naive" and "psychological" features as per Plonsky, Erev, Hazan, and Tennenholtz, 2017
    DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
    DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)
    diffEV = (np.matrix.dot(DistB[:, 0], DistB[:, 1]) - np.matrix.dot(DistA[:, 0], DistA[:, 1]))
    diffSDs = (getSD(DistB[:, 0], DistB[:, 1]) - getSD(DistA[:, 0], DistA[:, 1]))
    MinA = DistA[0, 0]
    MinB = DistB[0, 0]
    diffMins = MinB - MinA
    nA = DistA.shape[0]  # num outcomes in A
    nB = DistB.shape[0]  # num outcomes in B
    MaxA = DistA[nA - 1, 0]
    MaxB = DistB[nB - 1, 0]
    diffMaxs = MaxB - MaxA

    diffUV = (np.matrix.dot(DistB[:, 0], np.repeat([1 / nB], nB))) - (np.matrix.dot(DistA[:, 0], np.repeat([1 / nA], nA)))
    if Amb == 1:
        ambiguous = True
    else:
        ambiguous = False

    MaxOutcome = max(MaxA, MaxB)
    SignMax = np.sign(MaxOutcome)
    if MinA == MinB:
        RatioMin = 1
    elif np.sign(MinA) == np.sign(MinB):
        RatioMin = min(abs(MinA), abs(MinB)) / max(abs(MinA), abs(MinB))
    else:
        RatioMin = 0

    Range = MaxOutcome - min(MinA, MinB)
    diffSignEV = (Range * np.matrix.dot(np.sign(DistB[:, 0]), DistB[:, 1]) -
                  Range * np.matrix.dot(np.sign(DistA[:, 0]), DistA[:, 1]))
    trivial = CPC15_isStochasticDom(DistA, DistB)
    whchdom = trivial['which'][0]
    Dom = 0
    if trivial['dom'][0] and whchdom == 'A':
        Dom = -1
    if trivial['dom'][0] and whchdom == 'B':
        Dom = 1
    BEVa = np.matrix.dot(DistA[:, 0], DistA[:, 1])
    if ambiguous:
        UEVb = np.matrix.dot(DistB[:, 0], np.repeat(1 / nB, nB))
        BEVb = (UEVb + BEVa + MinB) / 3
        pEstB = np.repeat([float(nB)], 1)  # estimation of probabilties in Amb
        t_SPminb = (BEVb - np.mean(DistB[1:nB + 1, 0])) / (MinB - np.mean(DistB[1:nB + 1, 0]))
        if t_SPminb < 0:
            pEstB[0] = 0
        elif t_SPminb > 1:
            pEstB[0] = 1
        else:
            pEstB[0] = t_SPminb
        pEstB = np.append(pEstB, np.repeat([(1 - pEstB[0]) / (nB - 1)], nB - 1))
    else:
        pEstB = DistB[:, 1]
        BEVb = np.matrix.dot(DistB[:, 0], pEstB)

    diffBEV0 = (BEVb - BEVa)
    BEVfb = (BEVb + (np.matrix.dot(DistB[:, 0], DistB[:, 1]))) / 2
    diffBEVfb = (BEVfb - BEVa)

    sampleDistB = np.column_stack((DistB[:, 0], pEstB))
    probsBetter = get_pBetter(DistA, sampleDistB, corr=1)
    pAbetter = probsBetter[0]
    pBbetter = probsBetter[1]
    pBbet_Unbiased1 = pBbetter - pAbetter

    sampleUniDistA = np.column_stack((DistA[:, 0], np.repeat([1 / nA], nA)))
    sampleUniDistB = np.column_stack((DistB[:, 0], np.repeat([1 / nB], nB)))
    probsBetterUni = get_pBetter(sampleUniDistA, sampleUniDistB, corr=1)
    pBbet_Uniform = probsBetterUni[1] - probsBetterUni[0]

    sampleSignA = np.copy(DistA)
    sampleSignA[:, 0] = np.sign(sampleSignA[:, 0])
    sampleSignB = np.column_stack((np.sign(DistB[:, 0]), pEstB))
    probsBetterSign = get_pBetter(sampleSignA, sampleSignB, corr=1)
    pBbet_Sign1 = probsBetterSign[1] - probsBetterSign[0]
    sampleSignBFB = np.column_stack((np.sign(DistB[:, 0]), DistB[:, 1]))
    if Corr == 1:
        probsBetter = get_pBetter(DistA, DistB, corr=1)
        probsBetterSign = get_pBetter(sampleSignA, sampleSignBFB, corr=1)
    elif Corr == -1:
        probsBetter = get_pBetter(DistA, DistB, corr=-1)
        probsBetterSign = get_pBetter(sampleSignA, sampleSignBFB, corr=-1)
    else:
        probsBetter = get_pBetter(DistA, DistB, corr=0)
        probsBetterSign = get_pBetter(sampleSignA, sampleSignBFB, corr=0)

    pBbet_UnbiasedFB = probsBetter[1] - probsBetter[0]
    pBbet_SignFB = probsBetterSign[1] - probsBetterSign[0]

    # create features data frame
    feats_labels = ('diffEV', 'diffSDs', 'diffMins', 'diffMaxs', 'diffUV', 'RatioMin', 'SignMax',
                    'pBbet_Unbiased1', 'pBbet_UnbiasedFB', 'pBbet_Uniform', 'pBbet_Sign1', 'pBbet_SignFB', 'Dom',
                    'diffBEV0', 'diffBEVfb', 'diffSignEV')
    data_lists = [[diffEV, diffSDs, diffMins, diffMaxs, diffUV, RatioMin,
                   SignMax, pBbet_Unbiased1,pBbet_UnbiasedFB, pBbet_Uniform,
                   pBbet_Sign1, pBbet_SignFB, Dom, diffBEV0,diffBEVfb, diffSignEV]]
    features_data = [item for sublist in data_lists for item in sublist]
    tmpFeats = pd.DataFrame(features_data, index=feats_labels).T

    return tmpFeats
def Button(Button):
    if Button=='R':
        return 'L'
    else:
        return 'R'
def Outcomes(lot_shape,pH):
    if lot_shape=='_':
        if pH==1:
            return 1
        else:
            return 2
    else:
        return 3
def to_one_hot(value,category):
    if value==category:
        return 1
    else:
        return 0
def is_good(Bpay,Apay):
    if Bpay>Apay:
        return 1
    else:
        return 0
def value_trial(sequence,i):
    return sequence[i]
def append(array,val):
    array.append(val)
    return array
def feature_extration():
    df=pd.read_csv('All_estimation_raw_data.csv')
    df=df.drop(columns=['Set','Condition','RT','Feedback','block','Payoff','Forgone'])
    #adds opposite games for non amb games
    non_amb=df[df['Amb']==0]
    opposite=non_amb[['SubjID','Location','Gender','Age','Amb','Corr','Order','Trial']]
    opposite['GameID']=non_amb['GameID']+210
    opposite['Ha']=non_amb['Hb']
    opposite['pHa']=non_amb['pHb']
    opposite['La']=non_amb['Lb']
    opposite['LotShapeA']=non_amb['LotShapeB']
    opposite['LotNumA']=non_amb['LotNumB']
    opposite['Hb']=non_amb['Ha']
    opposite['pHb']=non_amb['pHa']
    opposite['Lb']=non_amb['La']
    opposite['LotShapeB']=non_amb['LotShapeA']
    opposite['LotNumB']=non_amb['LotNumA']
    opposite['Button']=non_amb.apply(lambda x: Button(x['Button']),axis=1)
    opposite['B']=1-non_amb['B']
    opposite['Apay']=non_amb['Bpay']
    opposite['Bpay']=non_amb['Apay']
    df=df.append(opposite)
    #adds category of num of outcomes
    df['a']=df.apply(lambda x: Outcomes(x['LotShapeA'],x['pHa']),axis=1)
    df['b']=df.apply(lambda x: Outcomes(x['LotShapeB'],x['pHb']),axis=1)
    #adds luck level per game,trial,and participent comnination
    Data=df[['GameID','Ha','pHa','La','LotShapeA','LotNumA','Hb','pHb','Lb','LotShapeB','LotNumB','Amb','Corr','Apay','Bpay']].drop_duplicates()
    nProblems = Data.shape[0]
    Data.index=range(nProblems)
    luck_levels_A=[]
    luck_levels_B=[]
    for prob in range(nProblems):
        Ha = Data['Ha'][prob]
        pHa = Data['pHa'][prob]
        La = Data['La'][prob]
        LotShapeA = Data['LotShapeA'][prob]
        LotNumA = Data['LotNumA'][prob]
        Hb = Data['Hb'][prob]
        pHb = Data['pHb'][prob]
        Lb = Data['Lb'][prob]
        LotShapeB = Data['LotShapeB'][prob]
        LotNumB = Data['LotNumB'][prob]
        Amb = Data['Amb'][prob]
        Corr = Data['Corr'][prob]
        Apay=Data['Apay'][prob]
        Bpay=Data['Bpay'][prob]
        luck_levels_A.append(luck_level(Ha,pHa,La,LotShapeA,LotNumA,Apay))
        luck_levels_B.append(luck_level(Hb,pHb,Lb,LotShapeB,LotNumB,Bpay))
    Data['luck_level_A']=luck_levels_A
    Data['luck_level_B']=luck_levels_B
    df=df.merge(Data[['GameID','luck_level_A','luck_level_B','Apay','Bpay']],on=['GameID','Apay','Bpay'])
    #adds biases choises per game and trial
    unbiased,uniform,pessimism,sign,GameID,Trial=[],[],[],[],[],[]
    Data=df[['GameID','Ha','pHa','La','LotShapeA','LotNumA','Hb','pHb','Lb','LotShapeB','LotNumB','Amb','Corr']].drop_duplicates()
    nProblems = Data.shape[0]
    Data.index=range(nProblems)
    for prob in range(nProblems):
        Ha = Data['Ha'][prob]
        pHa = Data['pHa'][prob]
        La = Data['La'][prob]
        LotShapeA = Data['LotShapeA'][prob]
        LotNumA = Data['LotNumA'][prob]
        Hb = Data['Hb'][prob]
        pHb = Data['pHb'][prob]
        Lb = Data['Lb'][prob]
        LotShapeB = Data['LotShapeB'][prob]
        LotNumB = Data['LotNumB'][prob]
        Amb = Data['Amb'][prob]
        Corr = Data['Corr'][prob]
        simPred_unbiased,simPred_uniform,simPred_pessimism,simPred_sign=CPC15_BEASTsimulation(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)
        unbiased.append(simPred_unbiased)
        uniform.append(simPred_uniform)
        pessimism.append(simPred_pessimism)
        sign.append(simPred_sign)
        GameID.append([Data['GameID'][prob] for i in range(25)])
        Trial.append([i+1 for i in range(25)])
    unbiased_list=[]
    uniform_list=[]
    pessimism_list=[]
    sign_list=[]
    GameID_list=[]
    Trial_list=[]
    unbiased_list=[]
    uniform_list=[]
    pessimism_list=[]
    sign_list=[]
    GameID_list=[]
    Trial_list=[]
    for i in range(len(unbiased)):
        for j in range(25):
            unbiased_list.append(unbiased[i][j][0])
            uniform_list.append(uniform[i][j][0])
            pessimism_list.append(pessimism[i][j][0])
            sign_list.append(sign[i][j][0])
            GameID_list.append(GameID[i][j])
            Trial_list.append(Trial[i][j])
    tmp=pd.DataFrame.from_dict({'Unbiased':unbiased_list,'Uniform':uniform_list,'Pessimism':pessimism_list,'Sign':sign_list,'GameID':GameID_list,'Trial':Trial_list})
    grouped=tmp.groupby(['GameID']).agg({'Sign':lambda x:list(x),'Unbiased':lambda x:list(x),'Uniform':lambda x:list(x),'Pessimism':lambda x:list(x)})
    for i  in range(25):
        for column in ['Unbiased','Uniform','Pessimism','Sign']:
            grouped[column+'_'+str(i+1)]=grouped.apply(lambda x: value_trial(x[column],i),axis=1)
    for column in ['Unbiased','Uniform','Pessimism','Sign']:
        grouped[column+'_Mean']=grouped[column].apply(lambda x: np.mean(np.array(x)))
        grouped[column+'_Var']=grouped[column].apply(lambda x: np.var(np.array(x)))
    grouped=grouped.drop(columns=['Unbiased','Uniform','Pessimism','Sign'])
    df=grouped.merge(df,on=['GameID'])
    #adds psychological features
    df2=df[df['GameID']<211].merge(pd.read_csv('RealData270.csv')[['GameID','diffEV', 'diffSDs', 'diffMins', 'diffMaxs', 'diffUV', 'RatioMin', 'SignMax','pBbet_Unbiased1', 'pBbet_UnbiasedFB', 'pBbet_Uniform', 'pBbet_Sign1', 'pBbet_SignFB', 'Dom','diffBEV0', 'diffBEVfb', 'diffSignEV']],on='GameID').sort_values(by=['SubjID', 'Order','GameID','Trial'])
    Data=df[df['GameID']>210][['GameID','Ha','pHa','La','LotShapeA','LotNumA','Hb','pHb','Lb','LotShapeB','LotNumB','Amb','Corr']].drop_duplicates()
    nProblems = Data.shape[0]
    Data.index=range(nProblems)
    for prob in range(nProblems):
        Ha = Data['Ha'][prob]
        pHa = Data['pHa'][prob]
        La = Data['La'][prob]
        LotShapeA = Data['LotShapeA'][prob]
        LotNumA = Data['LotNumA'][prob]
        Hb = Data['Hb'][prob]
        pHb = Data['pHb'][prob]
        Lb = Data['Lb'][prob]
        LotShapeB = Data['LotShapeB'][prob]
        LotNumB = Data['LotNumB'][prob]
        Amb = Data['Amb'][prob]
        Corr = Data['Corr'][prob]
        Feats=get_PF_Features(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)
        Feats['GameID']=Data['GameID'][prob]
        if prob==0:
            df3=Feats
        else:
            df3=df3.append(Feats)
    df3=df.merge(df3,on='GameID')
    df=df2.append(df3)
    #change from catgorial to one-hot
    df['lot_shape__A']=df.apply(lambda x: to_one_hot(x['LotShapeA'],'-'),axis=1)
    df['lot_shape_symm_A']=df.apply(lambda x: to_one_hot(x['LotShapeA'],'Symm'),axis=1)
    df['lot_shape_R_A']=df.apply(lambda x: to_one_hot(x['LotShapeA'],'R-skew'),axis=1)
    df['lot_shape_L_A']=df.apply(lambda x: to_one_hot(x['LotShapeA'],'L-skew'),axis=1)
    df['lot_shape__B']=df.apply(lambda x: to_one_hot(x['LotShapeB'],'-'),axis=1)
    df['lot_shape_symm_B']=df.apply(lambda x: to_one_hot(x['LotShapeB'],'Symm'),axis=1)
    df['lot_shape_R_B']=df.apply(lambda x: to_one_hot(x['LotShapeB'],'R-skew'),axis=1)
    df['lot_shape_L_B']=df.apply(lambda x: to_one_hot(x['LotShapeB'],'L-skew'),axis=1)
    df['R']=df.apply(lambda x: to_one_hot(x['Button'],'R'),axis=1)
    df['Technion']=df.apply(lambda x: to_one_hot(x['Location'],'Technion'),axis=1)
    df['M']=df.apply(lambda x: to_one_hot(x['Gender'],'M'),axis=1)
    df=df.drop(columns=['Button','Location','LotShapeA','LotShapeB','Gender'])
    #adds weather B is good
    df['Good']=df.apply(lambda x: is_good(x['Bpay'],x['Apay']),axis=1)
    final=df[df['Trial']<6].drop(columns=['luck_level_A','luck_level_B'])
    for i in range(-50,257):
        for val in ['Apay','Bpay']:
            df[val+'_'+str(i)]=0
    for trial2 in range(6,26):
        df['Good_'+str(trial2)]=0.5
    """
    old=pd.read_csv('22.csv')
    for col in ['Apay_List','Bpay_List','luck_level_A_List','luck_level_B_List']:
        old[col]=old.apply(lambda x: [float(x) for x in x[col][1:len(x[col])-1].split(', ')],axis=1)
    """
    for trial in range(6,26):
        print(trial)
        relevant=df[df['Trial']==trial]
        length=len(relevant)
        relevant.index=range(length)
        for val in ['Apay','Bpay']:
            pays=relevant[val]
            for value in range(-50,257):
                if trial==6:
                    tmp=np.zeros(length)
                else:
                    tmp=old[val+'_'+str(value)].values
                for i in range(length):
                    if value==pays[i]:
                        tmp[i]+=1
                relevant[val+'_'+str(value)]=tmp
            if trial==6:
                relevant[val+'_List']=relevant.apply(lambda x: [x[val]],axis=1)
            else:
                relevant[val+'_List']=old[val+'_List']
                relevant[val+'_List']=relevant.apply(lambda x: append(x[val+'_List'],x[val]),axis=1)
            relevant[val+'_Mean']=relevant.apply(lambda x: np.mean(x[val+'_List']),axis=1)
            relevant[val+'_Var']=relevant.apply(lambda x: np.var(x[val+'_List']),axis=1)
        for val in ['luck_level_A','luck_level_B']:
            if trial==6:
                relevant[val+'_List']=relevant.apply(lambda x: [x[val]],axis=1)
            else:
                relevant[val+'_List']=old[val+'_List']
                relevant[val+'_List']=relevant.apply(lambda x: append(x[val+'_List'],x[val]),axis=1)
            relevant['luck_level_Mean']=relevant.apply(lambda x: np.mean(x[val+'_List']),axis=1)
        for trial2 in range(6,trial+1):
                if trial==trial2:
                    relevant['Good_'+str(trial2)]=relevant['Good']
                else:
                    relevant['Good_'+str(trial2)]=old['Good']
        old=relevant
        old.to_csv(str(trial)+'.csv',index=False)
        relevant=relevant.drop(columns=['luck_level_A','luck_level_B','luck_level_A_List','luck_level_B_List','Apay_List','Bpay_List'])
        final=final.append(relevant)
    final=final.sort_values(by=['SubjID', 'Order','GameID','Trial']).drop_duplicates()
    final.to_csv('final.csv',index=False)



def luck_level(H, pH, L, LotShape, LotNum, payoff):
    Dist = CPC18_getDist(H, pH, L, LotShape, LotNum)
    numbers=Dist[:,0]
    probabilities=Dist[:,1]
    # Sampling a single number from a discrete distribution
    #   The possible Numbers in the distribution with their resective
    #   Probabilities. rndNum is a randomly drawn probability
    #
    #   Conditions on Input (not checked):
    #   1. Numbers and Probabilites correspond one to one (i.e. first number is
    #   drawn w.p. first probability etc). These are numpy arrays.
    #   2. rndNum is a number between zero and one.
    #   3. Probabilites is a probability vector (numpy array)
    # The output is a number (float)

    cum_prob = 0
    sampled_int = 0
    while payoff >=numbers[sampled_int]:
        cum_prob += probabilities[sampled_int]
        if payoff==numbers[sampled_int]:
            break
        else:
            sampled_int += 1
    return cum_prob

def CPC15_BEASTsimulation(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
    DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
    DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)
    # Simulation of the BEAST model.
    #  Input: 2 discrete distributions which are set as matrices of 1st column
    # as outcome and 2nd its probability. DistA and DistB are numpy matrices;
    #  Amb is a number 1 or 0, this is the ambiguous between the A and B.
    #  Corr is thw correlation between A and B, this is a number between -1 to 1.
    # Output: numpy array of zise: (nBlocks, 1)

    SIGMA = 7
    KAPA = 3
    BETA = 2.6
    GAMA = 0.5
    PSI = 0.07
    THETA = 1

    nTrials = 25
    firstFeedback = 6
    nBlocks = 5

    # draw personal traits
    sigma = SIGMA * np.random.uniform(size=1)
    kapa = np.random.choice(range(1, KAPA+1), 1)
    kapa=[100]
    beta = BETA * np.random.uniform(size=1)
    gama = GAMA * np.random.uniform(size=1)
    psi = PSI * np.random.uniform(size=1)
    theta = THETA * np.random.uniform(size=1)

    ObsPay = np.zeros(shape=(nTrials - firstFeedback + 1, 2))  # observed outcomes in A (col1) and B (col2)
    Decision_unbiased = np.empty(shape=(nTrials, 1))
    Decision_uniform = np.empty(shape=(nTrials, 1))
    Decision_pessimism = np.empty(shape=(nTrials, 1))
    Decision_sign = np.empty(shape=(nTrials, 1))
    simPred_unbiased = np.empty(shape=(nBlocks, 1))
    simPred_uniform = np.empty(shape=(nBlocks, 1))
    simPred_pessimism = np.empty(shape=(nBlocks, 1))
    simPred_sign = np.empty(shape=(nBlocks, 1))

    # Useful variables
    nA = DistA.shape[0]  # num outcomes in A
    nB = DistB.shape[0]  # num outcomes in B

    if Amb == 1:
        ambiguous = True
    else:
        ambiguous = False

    nfeed = 0  # "t"; number of outcomes with feedback so far
    pBias = np.array([beta / (beta + 1 + pow(nfeed, theta))])
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
    trivial = CPC15_isStochasticDom(DistA, DistB)
    BEVa = np.matrix.dot(DistA[:, 0], DistA[:, 1])
    if ambiguous:
        UEVb = np.matrix.dot(DistB[:, 0], np.repeat([1 / nB], nB))
        BEVb = (1-psi) * (UEVb+BEVa) / 2 + psi * MinB
        pEstB = np.repeat([float(nB)], 1)  # estimation of probabilties in Amb
        t_SPminb = (BEVb - np.mean(DistB[1:nB+1, 0])) / (MinB - np.mean(DistB[1:nB+1, 0]))
        if t_SPminb < 0:
            pEstB[0] = 0
        elif t_SPminb > 1:
            pEstB[0] = 1
        else:
            pEstB[0] = t_SPminb

        # Add nb-1 rows to pEstB:
        pEstB = np.append(pEstB, np.repeat((1 - pEstB[0]) / (nB - 1), nB-1))

    else:
        pEstB = DistB[:, 1]
        BEVb = np.matrix.dot(DistB[:, 0], pEstB)
    block=0
    # simulation of decisions
    for trial in range(nTrials):
        STa_unbiased = 0
        STb_unbiased = 0
        STa_uniform= 0
        STb_uniform = 0
        STa_pessimism = 0
        STb_pessimism = 0
        STa_sign = 0
        STb_sign = 0
        # mental simulations
        for s in range(1, kapa[0]+1):
            rndNum = np.random.uniform(size=2)
            if nfeed == 0:
                outcomeA_unbiased = distSample(DistA[:, 0], DistA[:, 1], rndNum[1])
                outcomeB_unbiased = distSample(DistB[:, 0], pEstB, rndNum[1])
            else:
                uniprobs = np.repeat([1 / nfeed], nfeed)
                outcomeA_unbiased = distSample(ObsPay[0:nfeed, 0], uniprobs, rndNum[1])
                outcomeB_unbiased = distSample(ObsPay[0:nfeed, 1], uniprobs, rndNum[1])
            outcomeA_uniform = distSample(DistA[:, 0], np.repeat([1 / nA], nA), rndNum[1])
            outcomeB_uniform = distSample(DistB[:, 0], np.repeat([1 / nB], nB), rndNum[1])
            if SignMax > 0 and RatioMin < gama:
                outcomeA_pessimism = MinA
                outcomeB_pessimism = MinB
            else:
                outcomeA_pessimism = distSample(DistA[:, 0], np.repeat([1 / nA], nA), rndNum[1])
                outcomeB_pessimism = distSample(DistB[:, 0], np.repeat([1 / nB], nB), rndNum[1])
            if nfeed == 0:
                outcomeA_sign = Range * distSample(np.sign(DistA[:, 0]), DistA[:, 1], rndNum[1])
                outcomeB_sign = Range * distSample(np.sign(DistB[:, 0]), pEstB, rndNum[1])
            else:
                uniprobs = np.repeat([1 / nfeed], nfeed)
                outcomeA_sign = Range * distSample(np.sign(ObsPay[0:nfeed, 0]), uniprobs, rndNum[1])
                outcomeB_sign = Range * distSample(np.sign(ObsPay[0:nfeed, 1]), uniprobs, rndNum[1])

            STa_unbiased = STa_unbiased + outcomeA_unbiased
            STb_unbiased = STb_unbiased + outcomeB_unbiased
            STa_uniform = STa_uniform + outcomeA_uniform
            STb_uniform = STb_uniform + outcomeB_uniform
            STa_pessimism = STa_pessimism + outcomeA_pessimism
            STb_pessimism = STb_pessimism + outcomeB_pessimism
            STa_sign = STa_sign + outcomeA_sign
            STb_sign = STb_sign + outcomeB_sign

        STa_unbiased = STa_unbiased / kapa
        STb_unbiased = STb_unbiased / kapa
        STa_uniform = STa_uniform / kapa
        STb_uniform = STb_uniform / kapa
        STa_pessimism = STa_pessimism / kapa
        STb_pessimism = STb_pessimism / kapa
        STa_sign = STa_sign / kapa
        STb_sign = STb_sign / kapa

        # error term
        if trivial['dom'][0]:
            error = 0
        else:
            error = sigma * np.random.normal(size=1)  # positive values contribute to attraction to A

        # decision
        Decision_unbiased[trial] = (BEVa - BEVb) + (STa_unbiased - STb_unbiased) + error < 0
        if (BEVa - BEVb) + (STa_unbiased - STb_unbiased) + error == 0:
            Decision_unbiased[trial] = np.random.choice(range(1, 3), size=1, replace=False) - 1
        Decision_uniform[trial] = (BEVa - BEVb) + (STa_uniform - STb_uniform) + error < 0
        if (BEVa - BEVb) + (STa_uniform - STb_uniform) + error == 0:
            Decision_uniform[trial] = np.random.choice(range(1, 3), size=1, replace=False) - 1
        Decision_pessimism[trial] = (BEVa - BEVb) + (STa_pessimism - STb_pessimism) + error < 0
        if (BEVa - BEVb) + (STa_pessimism - STb_pessimism) + error == 0:
            Decision_pessimism[trial] = np.random.choice(range(1, 3), size=1, replace=False) - 1
        Decision_sign[trial] = (BEVa - BEVb) + (STa_sign - STb_sign) + error < 0
        if (BEVa - BEVb) + (STa_sign - STb_sign) + error == 0:
            Decision_sign[trial] = np.random.choice(range(1, 3), size=1, replace=False) - 1

        if trial >= firstFeedback - 1:
            #  got feedback
            nfeed += 1
            pBias = np.append(pBias, beta / (beta + 1 + pow(nfeed, theta)))
            rndNumObs = np.random.uniform(size=1)
            ObsPay[nfeed - 1, 0] = distSample(DistA[:, 0], DistA[:, 1], rndNumObs)  # draw outcome from A
            if Corr == 1:
                ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], rndNumObs)
            elif Corr == -1:
                ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], 1-rndNumObs)
            else:
                # draw outcome from B
                ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], np.random.uniform(size=1))
            if ambiguous:
                BEVb = (1 - 1 / (nTrials-firstFeedback+1)) * BEVb + 1 / (nTrials-firstFeedback+1) * ObsPay[nfeed - 1, 1]

    return Decision_unbiased,Decision_uniform,Decision_pessimism,Decision_sign


def to_B1(train):
    df2 = train[train['block'] == 1].drop(columns=['BEASTpred', 'B_rate', 'block', 'Feedback'])
    df2['B.1'] = list(train[train['block'] == 1]['B_rate'])
    df2['B.2'] = list(train[train['block'] == 2]['B_rate'])
    df2['B.3'] = list(train[train['block'] == 3]['B_rate'])
    df2['B.4'] = list(train[train['block'] == 4]['B_rate'])
    df2['B.5'] = list(train[train['block'] == 5]['B_rate'])
    df2['BEASTpred.1'] = list(train[train['block'] == 1]['BEASTpred'])
    df2['BEASTpred.2'] = list(train[train['block'] == 2]['BEASTpred'])
    df2['BEASTpred.3'] = list(train[train['block'] == 3]['BEASTpred'])
    df2['BEASTpred.4'] = list(train[train['block'] == 4]['BEASTpred'])
    df2['BEASTpred.5'] = list(train[train['block'] == 5]['BEASTpred'])
    return df2
def logistic(B,X):
    return 1/(1+np.exp(-B*X))

def calc_weights(B,BEVb,BEVa,val_dist,probs,Corr):
    kapa={}
    for k1 in ['unb','uni','pes','sig']:
        kapa[k1]=[0,0]
        for k2 in ['unb','uni','pes','sig']:
            kapa[k1+'_'+k2]=[0,0]
            for k3 in ['unb','uni','pes','sig']:
                kapa[k1+'_'+k2+'_'+k3]=[0,0]
    for k1 in ['unb','uni','pes','sig']:
        val_A1,dist_A1,val_B1,dist_B1=val_dist[k1][0],val_dist[k1][1],val_dist[k1][2],val_dist[k1][3]
        for i1 in range(val_A1.shape[0]):
            for j1 in range(val_B1.shape[0]):
                if min(dist_A1[i1+1],dist_B1[j1+1])>max(dist_A1[i1],dist_B1[j1]):
                    kapa[k1][0]+=(min(dist_A1[i1+1],dist_B1[j1+1])-max(dist_A1[i1],dist_B1[j1]))*logistic(B,BEVb-BEVa+val_B1[j1]-val_A1[i1])
                    for k2 in ['unb','uni','pes','sig']:
                        val_A2,dist_A2,val_B2,dist_B2=val_dist[k2][0],val_dist[k2][1],val_dist[k2][2],val_dist[k2][3]
                        for i2 in range(val_A2.shape[0]):
                            for j2 in range(val_B2.shape[0]):
                                if min(dist_A2[i2+1],dist_B2[j2+1])>max(dist_A2[i2],dist_B2[j2]):
                                    kapa[k1+'_'+k2][0]+=(min(dist_A1[i1+1],dist_B1[j1+1])-max(dist_A1[i1],dist_B1[j1]))*(min(dist_A2[i2+1],dist_B2[j2+1])-max(dist_A2[i2],dist_B2[j2]))*logistic(B,BEVb-BEVa+(val_B1[j1]+val_B2[j2])/2-(val_A1[i1]+val_A2[i2])/2)
                                    for k3 in ['unb','uni','pes','sig']:
                                        val_A3,dist_A3,val_B3,dist_B3=val_dist[k3][0],val_dist[k3][1],val_dist[k3][2],val_dist[k3][3]
                                        for i3 in range(val_A3.shape[0]):
                                            for j3 in range(val_B3.shape[0]):
                                                if min(dist_A3[i3+1],dist_B3[j3+1])>max(dist_A3[i3],dist_B3[j3]):
                                                    kapa[k1+'_'+k2+'_'+k3][0]+=(min(dist_A1[i1+1],dist_B1[j1+1])-max(dist_A1[i1],dist_B1[j1]))*(min(dist_A2[i2+1],dist_B2[j2+1])-max(dist_A2[i2],dist_B2[j2]))*(min(dist_A3[i3+1],dist_B3[j3+1])-max(dist_A3[i3],dist_B3[j3]))*logistic(B,BEVb-BEVa+(val_B1[j1]+val_B2[j2]+val_B3[j3])/3-(val_A1[i1]+val_A2[i2]+val_A3[i3])/3)
                if Corr==-1:
                    if min(dist_A1[i1+1],1-dist_B1[j1])>max(dist_A1[i1],1-dist_B1[j1+1]):
                        kapa[k1][1]+=(min(dist_A1[i1+1],1-dist_B1[j1])-max(dist_A1[i1],1-dist_B1[j1+1]))*logistic(B,BEVb-BEVa+val_B1[j1]-val_A1[i1])
                        for k2 in ['unb','uni','pes','sig']:
                            val_A2,dist_A2,val_B2,dist_B2=val_dist[k2][0],val_dist[k2][1],val_dist[k2][2],val_dist[k2][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if min(dist_A2[i2+1],1-dist_B2[j2])>max(dist_A2[i2],1-dist_B2[j2+1]):
                                        kapa[k1+'_'+k2][1]+=(min(dist_A1[i1+1],1-dist_B1[j1])-max(dist_A1[i1],1-dist_B1[j1+1]))*(min(dist_A2[i2+1],1-dist_B2[j2])-max(dist_A2[i2],1-dist_B2[j2+1]))*logistic(B,BEVb-BEVa+(val_B1[j1]+val_B2[j2])/2-(val_A1[i1]+val_A2[i2])/2)
                                        for k3 in ['unb','uni','pes','sig']:
                                            val_A3,dist_A3,val_B3,dist_B3=val_dist[k3][0],val_dist[k3][1],val_dist[k3][2],val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3+1],1-dist_B3[j3])>max(dist_A3[i3],1-dist_B3[j3+1]):
                                                        kapa[k1+'_'+k2+'_'+k3][1]+=(min(dist_A1[i1+1],1-dist_B1[j1])-max(dist_A1[i1],1-dist_B1[j1+1]))*(min(dist_A2[i2+1],1-dist_B2[j2])-max(dist_A2[i2],1-dist_B2[j2+1]))*(min(dist_A3[i3+1],1-dist_B3[j3])-max(dist_A3[i3],1-dist_B3[j3+1]))*logistic(B,BEVb-BEVa+(val_B1[j1]+val_B2[j2]+val_B3[j3])/3-(val_A1[i1]+val_A2[i2]+val_A3[i3])/3)
                if Corr==0:
                    kapa[k1][1]+=(dist_A1[i1+1]-dist_A1[i1])*(dist_B1[j1+1]-dist_B1[j1])*logistic(B,BEVb-BEVa+val_B1[j1]-val_A1[i1])
                    for k2 in ['unb','uni','pes','sig']:
                        val_A2,dist_A2,val_B2,dist_B2=val_dist[k2][0],val_dist[k2][1],val_dist[k2][2],val_dist[k2][3]
                        for i2 in range(val_A2.shape[0]):
                            for j2 in range(val_B2.shape[0]):
                                kapa[k1+'_'+k2][1]+=(dist_A1[i1+1]-dist_A1[i1])*(dist_B1[j1+1]-dist_B1[j1])*(dist_A2[i2+1]-dist_A2[i2])*(dist_B2[j2+1]-dist_B2[j2])*logistic(B,BEVb-BEVa+(val_B1[j1]+val_B2[j2])/2-(val_A1[i1]+val_A2[i2])/2)
                                for k3 in ['unb','uni','pes','sig']:
                                    val_A3,dist_A3,val_B3,dist_B3=val_dist[k3][0],val_dist[k3][1],val_dist[k3][2],val_dist[k3][3]
                                    for i3 in range(val_A3.shape[0]):
                                        for j3 in range(val_B3.shape[0]):
                                            kapa[k1+'_'+k2+'_'+k3][1]+=(dist_A1[i1+1]-dist_A1[i1])*(dist_B1[j1+1]-dist_B1[j1])*(dist_A2[i2+1]-dist_A2[i2])*(dist_B2[j2+1]-dist_B2[j2])*(dist_A3[i3+1]-dist_A3[i3])*(dist_B3[j3+1]-dist_B3[j3])*logistic(B,BEVb-BEVa+(val_B1[j1]+val_B2[j2]+val_B3[j3])/3-(val_A1[i1]+val_A2[i2]+val_A3[i3])/3)
    for k1 in ['unb','uni','pes','sig']:
        probs[k1].append(kapa[k1][0]/3)
        for k2 in ['unb','uni','pes','sig']:
            probs[k1+'_'+k2].append(kapa[k1+'_'+k2][0]/3)
            for k3 in ['unb','uni','pes','sig']:
                 probs[k1+'_'+k2+'_'+k3].append(kapa[k1+'_'+k2+'_'+k3][0]/3)
    if Corr==1:
        for i in range(4):
            for k1 in ['unb','uni','pes','sig']:
                probs[k1].append(kapa[k1][0]/3)
                for k2 in ['unb','uni','pes','sig']:
                    probs[k1+'_'+k2].append(kapa[k1+'_'+k2][0]/3)
                    for k3 in ['unb','uni','pes','sig']:
                         probs[k1+'_'+k2+'_'+k3].append(kapa[k1+'_'+k2+'_'+k3][0]/3)
    else:
        for i in range(4):
            for k1 in ['unb','uni','pes','sig']:
                probs[k1].append(kapa[k1][1]/3)
                for k2 in ['unb','uni','pes','sig']:
                    probs[k1+'_'+k2].append(kapa[k1+'_'+k2][1]/3)
                    for k3 in ['unb','uni','pes','sig']:
                         probs[k1+'_'+k2+'_'+k3].append(kapa[k1+'_'+k2+'_'+k3][1]/3)
    return probs

def lot_shape_convert2(lot_shape):
    if (lot_shape == [1, 0, 0, 0]).all(): return '-'
    if (lot_shape == [0, 1, 0, 0]).all(): return 'Symm'
    if (lot_shape == [0, 0, 1, 0]).all(): return 'L-skew'
    if (lot_shape == [0, 0, 0, 1]).all(): return 'R-skew'

def MSE(x,df,label):
    return mean_squared_error(df[label],
    x[0]*df['unb']+
    x[0]*x[0]*df['unb_unb']+
    x[0]*x[0]*x[0]*df['unb_unb_unb']+
    x[0]*x[0]*x[1]*df['unb_unb_uni']+
    x[0]*x[0]*x[2]*df['unb_unb_pes']+
    x[0]*x[0]*x[3]*df['unb_unb_sig']+
    x[0]*x[1]*df['unb_uni']+
    x[0]*x[1]*x[0]*df['unb_uni_unb']+
    x[0]*x[1]*x[1]*df['unb_uni_uni']+
    x[0]*x[1]*x[2]*df['unb_uni_pes']+
    x[0]*x[1]*x[3]*df['unb_uni_sig']+
    x[0]*x[2]*df['unb_pes']+
    x[0]*x[2]*x[0]*df['unb_pes_unb']+
    x[0]*x[2]*x[1]*df['unb_pes_uni']+
    x[0]*x[2]*x[2]*df['unb_pes_pes']+
    x[0]*x[2]*x[3]*df['unb_pes_sig']+
    x[0]*x[3]*df['unb_sig']+
    x[0]*x[3]*x[0]*df['unb_sig_unb']+
    x[0]*x[3]*x[1]*df['unb_sig_uni']+
    x[0]*x[3]*x[2]*df['unb_sig_pes']+
    x[0]*x[3]*x[3]*df['unb_sig_sig']+
    x[1]*df['uni']+
    x[1]*x[0]*df['uni_unb']+
    x[1]*x[0]*x[0]*df['uni_unb_unb']+
    x[1]*x[0]*x[1]*df['uni_unb_uni']+
    x[1]*x[0]*x[2]*df['uni_unb_pes']+
    x[1]*x[0]*x[3]*df['uni_unb_sig']+
    x[1]*x[1]*df['uni_uni']+
    x[1]*x[1]*x[0]*df['uni_uni_unb']+
    x[1]*x[1]*x[1]*df['uni_uni_uni']+
    x[1]*x[1]*x[2]*df['uni_uni_pes']+
    x[1]*x[1]*x[3]*df['uni_uni_sig']+
    x[1]*x[2]*df['uni_pes']+
    x[1]*x[2]*x[0]*df['uni_pes_unb']+
    x[1]*x[2]*x[1]*df['uni_pes_uni']+
    x[1]*x[2]*x[2]*df['uni_pes_pes']+
    x[1]*x[2]*x[3]*df['uni_pes_sig']+
    x[1]*x[3]*df['uni_sig']+
    x[1]*x[3]*x[0]*df['uni_sig_unb']+
    x[1]*x[3]*x[1]*df['uni_sig_uni']+
    x[1]*x[3]*x[2]*df['uni_sig_pes']+
    x[1]*x[3]*x[3]*df['uni_sig_sig']+
    x[2]*df['pes']+
    x[2]*x[0]*df['pes_unb']+
    x[2]*x[0]*x[0]*df['pes_unb_unb']+
    x[2]*x[0]*x[1]*df['pes_unb_uni']+
    x[2]*x[0]*x[2]*df['pes_unb_pes']+
    x[2]*x[0]*x[3]*df['pes_unb_sig']+
    x[2]*x[1]*df['pes_uni']+
    x[2]*x[1]*x[0]*df['pes_uni_unb']+
    x[2]*x[1]*x[1]*df['pes_uni_uni']+
    x[2]*x[1]*x[2]*df['pes_uni_pes']+
    x[2]*x[1]*x[3]*df['pes_uni_sig']+
    x[2]*x[2]*df['pes_pes']+
    x[2]*x[2]*x[0]*df['pes_pes_unb']+
    x[2]*x[2]*x[1]*df['pes_pes_uni']+
    x[2]*x[2]*x[2]*df['pes_pes_pes']+
    x[2]*x[2]*x[3]*df['pes_pes_sig']+
    x[2]*x[3]*df['pes_sig']+
    x[2]*x[3]*x[0]*df['pes_sig_unb']+
    x[2]*x[3]*x[1]*df['pes_sig_uni']+
    x[2]*x[3]*x[2]*df['pes_sig_pes']+
    x[2]*x[3]*x[3]*df['pes_sig_sig']+
    x[3]*df['sig']+
    x[3]*x[0]*df['sig_unb']+
    x[3]*x[0]*x[0]*df['sig_unb_unb']+
    x[3]*x[0]*x[1]*df['sig_unb_uni']+
    x[3]*x[0]*x[2]*df['sig_unb_pes']+
    x[3]*x[0]*x[3]*df['sig_unb_sig']+
    x[3]*x[1]*df['sig_uni']+
    x[3]*x[1]*x[0]*df['sig_uni_unb']+
    x[3]*x[1]*x[1]*df['sig_uni_uni']+
    x[3]*x[1]*x[2]*df['sig_uni_pes']+
    x[3]*x[1]*x[3]*df['sig_uni_sig']+
    x[3]*x[2]*df['sig_pes']+
    x[3]*x[2]*x[0]*df['sig_pes_unb']+
    x[3]*x[2]*x[1]*df['sig_pes_uni']+
    x[3]*x[2]*x[2]*df['sig_pes_pes']+
    x[3]*x[2]*x[3]*df['sig_pes_sig']+
    x[3]*x[3]*df['sig_sig']+
    x[3]*x[3]*x[0]*df['sig_sig_unb']+
    x[3]*x[3]*x[1]*df['sig_sig_uni']+
    x[3]*x[3]*x[2]*df['sig_sig_pes']+
    x[3]*x[3]*x[3]*df['sig_sig_sig'])


def find_probs(train,label):
    x1 = minimize(fun=MSE, x0=np.array([0.25, 0.25, 0.25, 0.25]), args=(train[train['block'] == 1],label),
                  bounds=[(0., 1), (0., 1), (0., 1), (0., 1)],
                  constraints=({'type': 'eq', 'fun': lambda b: 1 - sum(b)},), method='slsqp').x
    x2 = minimize(fun=MSE, x0=np.array([0.25, 0.25, 0.25, 0.25]), args=(train[train['block'] == 2],label),
                  bounds=[(0., 1), (0., 1), (0., 1), (0., 1)],
                  constraints=({'type': 'eq', 'fun': lambda b: 1 - sum(b)},), method='slsqp').x
    x3 = minimize(fun=MSE, x0=np.array([0.25, 0.25, 0.25, 0.25]), args=(train[train['block'] == 3],label),
                  bounds=[(0., 1), (0., 1), (0., 1), (0., 1)],
                  constraints=({'type': 'eq', 'fun': lambda b: 1 - sum(b)},), method='slsqp').x
    x4 = minimize(fun=MSE, x0=np.array([0.25, 0.25, 0.25, 0.25]), args=(train[train['block'] == 4],label),
                  bounds=[(0., 1), (0., 1), (0., 1), (0., 1)],
                  constraints=({'type': 'eq', 'fun': lambda b: 1 - sum(b)},), method='slsqp').x
    x5 = minimize(fun=MSE, x0=np.array([0.25, 0.25, 0.25, 0.25]), args=(train[train['block'] == 5],label),
                  bounds=[(0., 1), (0., 1), (0., 1), (0., 1)],
                  constraints=({'type': 'eq', 'fun': lambda b: 1 - sum(b)},), method='slsqp').x
    return x1,x2,x3,x4,x5

def find_weights(Data,B,name):
    probs = {}
    for k1 in ['unb', 'uni', 'pes', 'sig']:
        probs[k1] = []
        for k2 in ['unb', 'uni', 'pes', 'sig']:
            probs[k1 + '_' + k2] = []
            for k3 in ['unb', 'uni', 'pes', 'sig']:
                probs[k1 + '_' + k2 + '_' + k3] = []
    Data = Data[Data['Amb'] == 0]
    nProblems = Data.shape[0]
    Data.index = range(nProblems)
    for prob in range(0, nProblems, 5):
        logging.info(str(prob/5))
        print(prob/5,datetime.now())
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
        DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
        DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)
        nA = DistA.shape[0]
        nB = DistB.shape[0]
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
        BEVb = np.matrix.dot(DistB[:, 0], DistB[:, 1])
        val_dist = {}
        val_dist['unb'] = [DistA[:, 0], np.concatenate(([0], np.cumsum(DistA[:, 1]))), DistB[:, 0],
                           np.concatenate(([0], np.cumsum(DistB[:, 1])))]
        val_dist['uni'] = [DistA[:, 0], np.concatenate(([0], np.cumsum(np.repeat([1 / nA], nA)))), DistB[:, 0],
                           np.concatenate(([0], np.cumsum(np.repeat([1 / nB], nB))))]
        if SignMax > 0 and RatioMin < 0.25:
            val_dist['pes'] = [np.array([MinA]), np.concatenate(([0], [1])), np.array([MinB]),
                               np.concatenate(([0], [1]))]
        else:
            val_dist['pes'] = [DistA[:, 0], np.concatenate(([0], np.cumsum(np.repeat([1 / nA], nA)))), DistB[:, 0],
                               np.concatenate(([0], np.cumsum(np.repeat([1 / nB], nB))))]
        val_dist['sig'] = [Range * np.sign(DistA[:, 0]), np.concatenate(([0], np.cumsum(DistA[:, 1]))),
                           Range * np.sign(DistB[:, 0]), np.concatenate(([0], np.cumsum(DistB[:, 1])))]
        probs = calc_weights(B, BEVb, BEVa, val_dist, probs, Corr)
        for key in probs.keys():
            if len(probs[key])!=prob+5:
                print(prob,len(probs[key]),key)
    for key in probs.keys():
        Data[key] = probs[key]
    Data.to_csv(name,index=False)
    return Data

if __name__ == '__main__':
    train=pd.read_csv('TrainDataWeights210.csv')
    test=pd.read_csv('TestDataWeights60.csv')
    #df=pd.read_csv('SyntheticDataWeights5000.csv')
    x1,x2,x3,x4,x5=find_probs(train,'B_rate')
    #print("synthetic BEASTpred")
    print("Train MSE")
    print(x1, MSE(x1, train[train['block'] == 1],'B_rate'))
    print(x2, MSE(x2, train[train['block'] == 2],'B_rate'))
    print(x3, MSE(x3, train[train['block'] == 3],'B_rate'))
    print(x4, MSE(x4, train[train['block'] == 4],'B_rate'))
    print(x5, MSE(x5, train[train['block'] == 5],'B_rate'))
    print("Test MSE")
    print(x1, MSE(x1, test[test['block'] == 1],'B_rate'))
    print(x2, MSE(x2, test[test['block'] == 2],'B_rate'))
    print(x3, MSE(x3, test[test['block'] == 3],'B_rate'))
    print(x4, MSE(x4, test[test['block'] == 4],'B_rate'))
    print(x5, MSE(x5, test[test['block'] == 5],'B_rate'))
    """
    data=train.append(test)
    # data=data[data['GameID']>30]
    # for i in range(5):
    # probs=np.random.choice(range(30,271), 48)
    # train=Data[~Data['GameID'].isin(probs)]
    # test=Data[Data['GameID'].isin(probs)]
    a1 = data[data['lot_shape__A'] == 1]
    a2 = a1[a1['pHa'] < 1]
    a1 = a1[a1['pHa'] == 1]
    a3 = data[data['lot_shape__A'] == 0]

    a1_b1 = a1[a1['lot_shape__B'] == 1]
    a1_b2 = a1_b1[a1_b1['pHb'] < 1]
    a1_b1 = a1_b1[a1_b1['pHb'] == 1]
    a1_b3 = a1[a1['lot_shape__B'] == 0]

    a2_b1 = a2[a2['lot_shape__B'] == 1]
    a2_b2 = a2_b1[a2_b1['pHb'] < 1]
    a2_b1 = a2_b1[a2_b1['pHb'] == 1]
    a2_b3 = a2[a2['lot_shape__B'] == 0]

    a3_b1 = a3[a3['lot_shape__B'] == 1]
    a3_b2 = a3_b1[a3_b1['pHb'] < 1]
    a3_b1 = a3_b1[a3_b1['pHb'] == 1]
    a3_b3 = a3[a3['lot_shape__B'] == 0]
    one_vs_one = a1_b1
    one_vs_two = a1_b2.append(a2_b1)
    one_vs_three = a1_b3.append(a3_b1)
    two_vs_two = a2_b2
    two_vs_three = a3_b2.append(a2_b3)
    three_vs_three = a3_b3

    names = ['one_vs_one', 'one_vs_two', 'one_vs_three', 'two_vs_two', 'two_vs_three', 'three_vs_three']
    dfs = [one_vs_one, one_vs_two, one_vs_three, two_vs_two, two_vs_three, three_vs_three]
    probas=['Punb','Puni','Ppes','Psig']
    for i in range(len(names)):
        df=dfs[i]
        leng=int(len(df)/5)
        print(names[i],leng)
        if leng>0:
            train=df[df['GameID']<211]
            test=df[df['GameID']>210]
            leng_train=int(len(train)/5)
            leng_test=int(len(test)/5)
            print(leng_train,leng_test)
            if len(train)>0 and len(test)>0:
                x1,x2,x3,x4,x5=find_probs(train,'BEASTpred')
                print("real BEASTpred")
                print(x1, MSE(x1, train[train['block'] == 1],'BEASTpred'))
                print(x2, MSE(x2, train[train['block'] == 2],'BEASTpred'))
                print(x3, MSE(x3, train[train['block'] == 3],'BEASTpred'))
                print(x4, MSE(x4, train[train['block'] == 4],'BEASTpred'))
                print(x5, MSE(x5, train[train['block'] == 5],'BEASTpred'))
                print("real B_rate")
                print(x1, MSE(x1, train[train['block'] == 1],'B_rate'))
                print(x2, MSE(x2, train[train['block'] == 2],'B_rate'))
                print(x3, MSE(x3, train[train['block'] == 3],'B_rate'))
                print(x4, MSE(x4, train[train['block'] == 4],'B_rate'))
                print(x5, MSE(x5, train[train['block'] == 5],'B_rate'))
                for i in range(4):
                    tmp=[]
                    for j in range(leng_train):
                        for x in [x1,x2,x3,x4,x5]:
                            tmp.append(x[i])
                    train[probas[i]]=tmp
                    tmp = []
                    for j in range(leng_test):
                        for x in [x1,x2,x3,x4,x5]:
                            tmp.append(x[i])
                    test[probas[i]]=tmp
                train2=create_syntetic_dataset(train,is_probs=True,original=False)
                test2 = create_syntetic_dataset(test,is_probs=True,original=False)
                print("train",mean_squared_error(train2['B_rate'],train2['BEASTpred']))
                print("test", mean_squared_error(test2['B_rate'], test2['BEASTpred']))
                x1,x2,x3,x4,x5=find_probs(train,'B_rate')
                print("real B_rate")
                print(x1, MSE(x1, train[train['block'] == 1],'B_rate'))
                print(x2, MSE(x2, train[train['block'] == 2],'B_rate'))
                print(x3, MSE(x3, train[train['block'] == 3],'B_rate'))
                print(x4, MSE(x4, train[train['block'] == 4],'B_rate'))
                print(x5, MSE(x5, train[train['block'] == 5],'B_rate'))
                for i in range(4):
                    tmp=[]
                    for j in range(leng_train):
                        for x in [x1,x2,x3,x4,x5]:
                            tmp.append(x[i])
                    train[probas[i]]=tmp
                    tmp = []
                    for j in range(leng_test):
                        for x in [x1,x2,x3,x4,x5]:
                            tmp.append(x[i])
                    test[probas[i]]=tmp
                train2=create_syntetic_dataset(train,is_probs=True,original=False)
                test2 = create_syntetic_dataset(test,is_probs=True,original=False)
                print("train",mean_squared_error(train2['B_rate'],train2['BEASTpred']))
                print("test", mean_squared_error(test2['B_rate'], test2['BEASTpred']))
    """



