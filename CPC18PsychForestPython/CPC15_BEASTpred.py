import numpy as np
from .CPC18_getDist import CPC18_getDist
from .CPC15_BEASTsimulation import CPC15_BEASTsimulation
from torch import distributions
from torch.nn import functional as F
import torch

def CPC15_BEASTpred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr,p1,p2,p3,p4,p5,original):
    # Prediction of (the original) BEAST model for one problem
    # Input: for a and b: high outcome (Ha/ Hb: int), its probability (pHa/ pHb: double), low outcome
    #  (La/ Lb: int), the shape of the lottery (LotShapeA/ LotShapeB that can be:'-'/'Symm'/'L-skew'/'R-skew' only),
    #  the number of outcomes in the lottery (lot_numA/ LotNumB: int),
    #  Amb indicates if B is ambiguous (=1) or not (=0).
    #  Corr is the correlation between A and B, this is a number between -1 to 1.
    # Output: is the prediction of the BEAST model: this is a numpy of size (5,1)

    Prediction = np.repeat([0], 5)
    Prediction.shape = (5, 1)

    # get both options' distributions
    DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
    DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)

    # run model simulation nSims times
    nSims = 1000
    for sim in range(0, nSims):
        simPred = CPC15_BEASTsimulation(DistA, DistB, Amb, Corr,p1,p2,p3,p4,p5,original)
        Prediction = np.add(Prediction, (1 / nSims) * simPred)

    return Prediction
