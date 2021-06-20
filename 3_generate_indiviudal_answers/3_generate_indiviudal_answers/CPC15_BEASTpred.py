import numpy as np
from CPC18_getDist import CPC18_getDist
from CPC15_BEASTsimulation import CPC15_BEASTsimulation
import os
import logging
import time



def CPC15_BEAST_individual_pred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
    # Prediction of (the original) BEAST model for one problem
    # Input: for a and b: high outcome (Ha/ Hb: int), its probability (pHa/ pHb: double), low outcome
    #  (La/ Lb: int), the shape of the lottery (LotShapeA/ LotShapeB that can be:'-'/'Symm'/'L-skew'/'R-skew' only),
    #  the number of outcomes in the lottery (lot_numA/ LotNumB: int),
    #  Amb indicates if B is ambiguous (=1) or not (=0).
    #  Corr is the correlation between A and B, this is a number between -1 to 1.
    # Output: is the prediction of the BEAST model: this is a numpy of size (5,1)
    
    
    avg_Prediction = np.repeat([0], 1) # change 5 to 1 => only 1 block 
    avg_Prediction.shape = (1, 1) # change 5 to 1 => only 1 block 
    
    num_types = 3 # 54
#     all_Prediction = [[]]*200*54  # change 5 to 1 => only 1 block 
    all_Prediction = [[]]*200*num_types  # change 5 to 1 => only 1 block 

    # get both options' distributions
    DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
    DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)

    # run model simulation nSims times
    nSims = 200
    for sim in range(0, nSims):
#         if sim % 100 == 0 : 
#             logging.info('{}: Finish sim number : {}'.format((time.asctime(time.localtime(time.time()))), sim))
        simPred = CPC15_BEASTsimulation(DistA, DistB, Amb, Corr)
        for t in range(num_types) :
            avg_Prediction = np.add(avg_Prediction, (1 / (nSims*num_types)) * simPred[t][0])
            simPred[t][0] = round((simPred[t][0])*5)/5
            all_Prediction[sim*num_types +t] = simPred[t]

        
    return all_Prediction, avg_Prediction
