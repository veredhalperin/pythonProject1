import math
import numpy as np
from .CPC18_getDist import CPC18_getDist
from .CPC15_isStochasticDom import CPC15_isStochasticDom
from .get_pBetter import get_pBetter
from .CPC15_BEASTpred import CPC15_BEASTpred
import pandas as pd


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

    # convert lot shape: '-'/'Symm'/'L-skew'/'R-skew' to 4 different features for the RF model
    lot_shape_listA = lot_shape_convert(LotShapeA)
    lot_shape_listB = lot_shape_convert(LotShapeB)

    # create features data frame
    feats_labels = ('Ha', 'pHa', 'La', 'lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A', 'LotNumA',
                    'Hb', 'pHb', 'Lb', 'lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B', 'LotNumB',
                    'Amb', 'Corr', 'diffEV', 'diffSDs', 'diffMins', 'diffMaxs', 'diffUV', 'RatioMin', 'SignMax',
                    'pBbet_Unbiased1', 'pBbet_UnbiasedFB', 'pBbet_Uniform', 'pBbet_Sign1', 'pBbet_SignFB', 'Dom',
                    'diffBEV0', 'diffBEVfb', 'diffSignEV')
    data_lists = [[Ha, pHa, La], lot_shape_listA, [LotNumA, Hb, pHb, Lb], lot_shape_listB, [LotNumB, Amb, Corr,
                             diffEV, diffSDs, diffMins, diffMaxs, diffUV, RatioMin, SignMax, pBbet_Unbiased1,
                             pBbet_UnbiasedFB, pBbet_Uniform, pBbet_Sign1, pBbet_SignFB, Dom, diffBEV0,
                             diffBEVfb, diffSignEV]]
    features_data = [item for sublist in data_lists for item in sublist]
    tmpFeats = pd.DataFrame(features_data, index=feats_labels).T

    # duplicate features data frame as per number of blocks
    Feats = pd.concat([tmpFeats] * 5)

    # get BEAST model prediction as feature
    beastPs = CPC15_BEASTpred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)
    Feats['BEASTpred'] = beastPs

    Feats['block'] = np.arange(1, 6)
    Feats['Feedback'] = 1
    Feats.loc[Feats['block'] == 1, 'Feedback'] = 0

    return Feats


# To compute the distribution's standard deviation
def getSD(vals, probs):
    m = np.matrix.dot(vals, probs.T)
    sqds = np.power((vals - m), 2)
    var = np.matrix.dot(probs, sqds.T)
    return math.sqrt(var)


# Convert lot shape feautre to vector of 4 features
def lot_shape_convert(lot_shape):
    return {
        '-': [1, 0, 0, 0],
        'Symm': [0, 1, 0, 0],
        'L-skew': [0, 0, 1, 0],
        'R-skew': [0, 0, 0, 1],
    }
