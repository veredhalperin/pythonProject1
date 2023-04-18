import os
#os.chdir('---')
#####################################################################################
### Section A: Please change this section to import necessary files and packages ###
#####################################################################################
import pandas as pd
import numpy as np
import time
from CPC18PsychForestPython.CPC15_BEASTpred import CPC15_BEASTpred

if __name__ == '__main__':
    ####################################################
    ### Section B: Please do not change this section ###
    ####################################################
    # load problems to predict (in this example, the estimation set problems)
    Data = pd.read_csv('synth.csv')
    # useful variables
    nProblems = Data.shape[0]
    PredictedAll = np.zeros(shape=(nProblems, 5))
    ### End of Section A ###

    #################################################################
    ### Section C: Please change only lines 41-47 in this section ###
    #################################################################
    B_rate=[]
    for prob in range(nProblems):
        print(prob)
        # read problem's parameters
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

        # please plug in here your model that takes as input the 12 parameters
        # defined above and gives as output a vector size 5 named "Prediction"
        # in which each cell is the predicted B-rate for one block of five trials
        # example:
        B_rate.append(CPC15_BEASTpred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)[0][0])
    Data['B_rate']=B_rate
    Data.to_csv('synth2.csv',index=False)
