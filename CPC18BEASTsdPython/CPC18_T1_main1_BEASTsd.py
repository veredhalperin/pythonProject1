import os
#os.chdir('---')
#####################################################################################
### Section A: Please change this section to import necessary files and packages ###
#####################################################################################
import pandas as pd
import numpy as np
import time
from .CPC18_BEASTsd_pred import CPC18_BEASTsd_pred
def lot_shape_convert2(lot_shape):
    if (lot_shape == [1, 0, 0, 0]).all(): return '-'
    if (lot_shape == [0, 1, 0, 0]).all(): return 'Symm'
    if (lot_shape == [0, 0, 1, 0]).all(): return 'L-skew'
    if (lot_shape == [0, 0, 0, 1]).all(): return 'R-skew'

if __name__ == '__main__':
    ####################################################
    ### Section B: Please do not change this section ###
    ####################################################
    # load problems to predict (in this example, the estimation set problems)
    Data = pd.read_csv('c13k_selections-1.csv')
    # useful variables
    nProblems = Data.shape[0]
    PredictedAll = np.zeros(shape=(nProblems, 5))
    ### End of Section A ###

    #################################################################
    ### Section C: Please change only lines 40-45 in this section ###
    #################################################################
    for prob in range(0, nProblems):
        # read problem's parameters
        Ha = Data['Ha'][prob]
        pHa = Data['pHa'][prob]
        La = Data['La'][prob]
        #LotShapeA = Data['LotShapeA'][prob]
        LotShapeA = lot_shape_convert2(
            Data[['lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A']].values[prob])
        LotNumA = Data['LotNumA'][prob]
        Hb = Data['Hb'][prob]
        pHb = Data['pHb'][prob]
        Lb = Data['Lb'][prob]
        #LotShapeB = Data['LotShapeB'][prob]
        LotShapeB = lot_shape_convert2(
            Data[['lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B']].values[prob])
        LotNumB = Data['LotNumB'][prob]
        Amb = Data['Amb'][prob]
        Corr = Data['Corr'][prob]

        # please plug in here your model that takes as input the 12 parameters
        # defined above and gives as output a vector size 5 named "Prediction"
        # in which each cell is the predicted B-rate for one block of five trials
        # example:
        Prediction = CPC18_BEASTsd_pred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)
        # end of example

        PredictedAll[prob, :] = Prediction
        # for verbose progression
        print('{}: Finish problem number: {}'.format((time.asctime(time.localtime(time.time()))), prob+1))

    ### End of Section C ###

    ####################################################
    ### Section D: Please do not change this section ###
    ####################################################
    # compute MSE
    ObservedAll = Data[['B.1', 'B.2', 'B.3', 'B.4', 'B.5']]
    probMSEs = 100 * ((PredictedAll - ObservedAll) ** 2).mean(axis=1)
    totalMSE = np.mean(probMSEs)
    print('MSE over the {} problems: {}'.format(nProblems, totalMSE))
    # for keeping the predicted choice rates
    np.savetxt("outputAll.csv", PredictedAll, delimiter=",", header = "B1,B2,B3,B4,B5", fmt='%.4f')

    ### End of Section D ###