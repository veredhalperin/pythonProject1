import numpy as np
from CPC15_isStochasticDom import CPC15_isStochasticDom
from distSample import distSample


def CPC15_BEASTsimulation(DistA, DistB, Amb, Corr):
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

    nTrials = 5
    firstFeedback = 6
    nBlocks = 1

    # draw personal traits
#     sigma = SIGMA * np.random.uniform(size=1)
#     kapa = np.random.choice(range(1, KAPA+1), 1)
#     beta = BETA * np.random.uniform(size=1)
#     gama = GAMA * np.random.uniform(size=1)
    psi = PSI * np.random.uniform(size=1)
    theta = THETA * np.random.uniform(size=1)
    wamb = 0
    
    ObsPay = np.zeros(shape=(nTrials - firstFeedback + 1, 2))  # observed outcomes in A (col1) and B (col2)

    # Useful variables
    nA = DistA.shape[0]  # num outcomes in A
    nB = DistB.shape[0]  # num outcomes in B

    ambiguous = False

    nfeed = 0  # "t"; number of outcomes with feedback so far
    
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

    return_all =[]
  # 54 Types of individuals : 
 #     for sigma in [float(round(0.5*7/3,2)), float(round(1.5*7/3,2)), float(round(2.5*7/3,2))]:
#         for beta in [float(round(0.5*2.6/3,2)), float(round(1.5*2.6/3, 2)), float(round(2.5*2.6/3, 2))] : 
#             for kapa in [1,2,3] : 
#                 for gama in [float(round(0.5/3,2)), float(round(2*0.5/3,2))] : 

# 3 types of Individual according to kapa : 
    sigma=float(round(1.5*7/3,2))
    beta = float(round(1.5*2.6/3, 2)) 
    gama = float(round(0.5/2,2))

    for kapa in [1,2,3] : 

        Decision = np.empty(shape=(nTrials, 1))
#                     simPred = np.empty(shape=(nBlocks, 1))

        pBias = np.array([beta / (beta + 1 + pow(nfeed, theta))])

        trivial = CPC15_isStochasticDom(DistA, DistB)

        # simulation of decisions
        for trial in range(nTrials):
            STa = 0
            STb = 0
            # mental simulations
            for s in range(1, kapa+1):
                rndNum = np.random.uniform(size=2)
                if rndNum[0] > pBias[nfeed]:  # Unbiased technique
                    if nfeed == 0:
                        outcomeA = distSample(DistA[:, 0], DistA[:, 1], rndNum[1])
                        outcomeB = distSample(DistB[:, 0], pEstB, rndNum[1])
                    else:
                        uniprobs = np.repeat([1 / nfeed], nfeed)
                        outcomeA = distSample(ObsPay[0:nfeed, 0], uniprobs, rndNum[1])
                        outcomeB = distSample(ObsPay[0:nfeed, 1], uniprobs, rndNum[1])

                elif rndNum[0] > (2 / 3) * pBias[nfeed]:  # uniform
                    outcomeA = distSample(DistA[:, 0], np.repeat([1 / nA], nA), rndNum[1])
                    outcomeB = distSample(DistB[:, 0], np.repeat([1 / nB], nB), rndNum[1])

                elif rndNum[0] > (1 / 3) * pBias[nfeed]:  # contingent pessimism
                    if SignMax > 0 and RatioMin < gama:
                        outcomeA = MinA
                        outcomeB = MinB
                    else:
                        outcomeA = distSample(DistA[:, 0], np.repeat([1 / nA], nA), rndNum[1])
                        outcomeB = distSample(DistB[:, 0], np.repeat([1 / nB], nB), rndNum[1])

                else:  # Sign
                    if nfeed == 0:
                        outcomeA = Range * distSample(np.sign(DistA[:, 0]), DistA[:, 1], rndNum[1])
                        outcomeB = Range * distSample(np.sign(DistB[:, 0]), pEstB, rndNum[1])
                    else:
                        uniprobs = np.repeat([1 / nfeed], nfeed)
                        outcomeA = Range * distSample(np.sign(ObsPay[0:nfeed, 0]), uniprobs, rndNum[1])
                        outcomeB = Range * distSample(np.sign(ObsPay[0:nfeed, 1]), uniprobs, rndNum[1])

                STa = STa + outcomeA
                STb = STb + outcomeB

            STa = STa / kapa
            STb = STb / kapa

            # error term
            if trivial['dom'][0]:
                error = 0
            else:
                error = sigma * np.random.normal(size=1)  # positive values contribute to attraction to A

            # decision
            Decision[trial] = (BEVa - BEVb) + (STa - STb) + error < 0
            if (BEVa - BEVb) + (STa - STb) + error == 0:
                Decision[trial] = np.random.choice(range(1, 3), size=1, replace=False) - 1

        # compute B-rates for this simulation

        simPred = np.mean(Decision[0:5])

        return_all.append([simPred, sigma, kapa, beta, gama, psi[0], theta[0], wamb] )
                       
    
    return return_all
