from scipy.optimize import minimize
from CPC18PsychForestPython.distSample import distSample
from CPC18PsychForestPython.CPC18_getDist import CPC18_getDist
import math
import numpy as np
from CPC18PsychForestPython.CPC15_isStochasticDom import CPC15_isStochasticDom
import pandas as pd
from sklearn.metrics import mean_squared_error


def lot_shape_convert(lot_shape):
    return {
        '-': [1, 0, 0, 0],
        'Symm': [0, 1, 0, 0],
        'L-skew': [0, 0, 1, 0],
        'R-skew': [0, 0, 0, 1],
    }[lot_shape]


def lot_shape_convert2(lot_shape):
    if (lot_shape == [1, 0, 0, 0]).all(): return '-'
    if (lot_shape == [0, 1, 0, 0]).all(): return 'Symm'
    if (lot_shape == [0, 0, 1, 0]).all(): return 'L-skew'
    if (lot_shape == [0, 0, 0, 1]).all(): return 'R-skew'


def getSD(vals, probs):
    m = np.matrix.dot(vals, probs.T)
    sqds = np.power((vals - m), 2)
    var = np.matrix.dot(probs, sqds.T)
    return math.sqrt(var)


def Outcomes(lot_shape, pH):
    if lot_shape == '_':
        if pH == 1:
            return 1
        else:
            return 2
    else:
        return 3


def one_hot(value, category):
    if value == category:
        return 1
    else:
        return 0


def is_good(Bpay, Apay):
    if Bpay > Apay:
        return 1
    else:
        return 0


def value_trial(sequence, i):
    return sequence[i]


def append(array, val):
    array.append(val)
    return array


def luck_level(H, pH, L, LotShape, LotNum, payoff):
    Dist = CPC18_getDist(H, pH, L, LotShape, LotNum)
    numbers = Dist[:, 0]
    probabilities = Dist[:, 1]
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
    while payoff >= numbers[sampled_int]:
        cum_prob += probabilities[sampled_int]
        if payoff == numbers[sampled_int]:
            break
        else:
            sampled_int += 1
    return cum_prob


def avg_difference(BEVa, BEVb, STa, STb):
    return np.mean(BEVb - BEVa + STb - STa)


def calc_avg_difference(Data):
    for i in range(99, 9801, 100):
        print(i)
        df = pd.read_csv(str(i) + '.csv')
        for j in range(5):
            df['avg_difference_' + str(j)] = df.apply(lambda x: avg_difference(x1['BEVa'], x1['BEVb'],
                                                                               np.array(
                                                                                   [x1['STa_' + str(i) + '_' + str(j)]
                                                                                    for i in range(4000)]),
                                                                               np.array(
                                                                                   [x1['STb_' + str(i) + '_' + str(j)]
                                                                                    for i in range(4000)]), ), axis=1)
        df['avg_difference'] = (df['avg_difference_0'] + df['avg_difference_1'] + df['avg_difference_2'] + df[
            'avg_difference_3'] + df['avg_difference_4']) / 5
        tmp = Data.merge(df[['Problem', 'avg_difference']], on=['Problem'])
        if i == 100:
            data = tmp
        else:
            data = data.append(tmp)
    return data


def CPC15_BEASTsimulation_biased(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
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
    kapa = np.random.choice(range(1, KAPA + 1), 1)
    kapa = [100]
    beta = BETA * np.random.uniform(size=1)
    gama = GAMA * np.random.uniform(size=1)
    psi = PSI * np.random.uniform(size=1)
    theta = THETA * np.random.uniform(size=1)

    ObsPay = np.zeros(shape=(nTrials - firstFeedback + 1, 2))  # observed outcomes in A (col1) and B (col2)
    Decision_unbiased = np.empty(shape=(nTrials, 1))
    Decision_uniform = np.empty(shape=(nTrials, 1))
    Decision_pessimism = np.empty(shape=(nTrials, 1))
    Decision_sign = np.empty(shape=(nTrials, 1))

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
        BEVb = (1 - psi) * (UEVb + BEVa) / 2 + psi * MinB
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

    else:
        pEstB = DistB[:, 1]
        BEVb = np.matrix.dot(DistB[:, 0], pEstB)
    block = 0
    # simulation of decisions
    for trial in range(nTrials):
        STa_unbiased = 0
        STb_unbiased = 0
        STa_uniform = 0
        STb_uniform = 0
        STa_pessimism = 0
        STb_pessimism = 0
        STa_sign = 0
        STb_sign = 0
        # mental simulations
        for s in range(1, kapa[0] + 1):
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
                ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], 1 - rndNumObs)
            else:
                # draw outcome from B
                ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], np.random.uniform(size=1))
            if ambiguous:
                BEVb = (1 - 1 / (nTrials - firstFeedback + 1)) * BEVb + 1 / (nTrials - firstFeedback + 1) * ObsPay[
                    nfeed - 1, 1]

    return Decision_unbiased, Decision_uniform, Decision_pessimism, Decision_sign


def add_opposite_games(df, is_individuals, baseline=None, per_game_and_player=False):
    non_amb = df[df['Amb'] == 0]
    if is_individuals:
        opposite = non_amb[['SubjID', 'Technion', 'M', 'Age', 'Amb', 'Corr', 'Order', 'Trial']]
    elif per_game_and_player:
        opposite = non_amb[['SubjID', 'Technion', 'M', 'Age', 'Amb', 'Corr', 'Order']]
    else:
        opposite = non_amb[['Amb', 'Corr', ]]
    opposite['GameID'] = non_amb['GameID'] + 270
    opposite['Ha'] = non_amb['Hb']
    opposite['pHa'] = non_amb['pHb']
    opposite['La'] = non_amb['Lb']
    for shape in ['lot_shape__', 'lot_shape_symm_', 'lot_shape_L_', 'lot_shape_R_']:
        opposite[shape + 'A'] = non_amb[shape + 'B']
    opposite['LotNumA'] = non_amb['LotNumB']
    opposite['Hb'] = non_amb['Ha']
    opposite['pHb'] = non_amb['pHa']
    opposite['Lb'] = non_amb['La']
    for shape in ['lot_shape__', 'lot_shape_symm_', 'lot_shape_L_', 'lot_shape_R_']:
        opposite[shape + 'B'] = non_amb[shape + 'A']
    opposite['LotNumB'] = non_amb['LotNumA']
    if is_individuals:
        opposite['R'] = non_amb.apply(lambda x: R(x1['R']), axis=1)
        opposite['B'] = 1 - non_amb['B']
        opposite['Apay'] = non_amb['Bpay']
        opposite['Bpay'] = non_amb['Apay']
    else:
        for i in range(1, 6):
            opposite['B.' + str(i)] = 1 - non_amb['B.' + str(i)]
            opposite[baseline + '.' + str(i)] = 1 - non_amb[baseline + '.' + str(i)]
            if per_game_and_player:
                opposite['Apay' + '.' + str(i)] = non_amb['Bpay' + '.' + str(i)]
                opposite['Bpay' + '.' + str(i)] = non_amb['Apay' + '.' + str(i)]
    return opposite


def change_to_one_hot(df, only_shapes=True):
    df['lot_shape__A'] = df.apply(lambda x: one_hot(x1['LotShapeA'], '-'), axis=1)
    df['lot_shape_symm_A'] = df.apply(lambda x: one_hot(x1['LotShapeA'], 'Symm'), axis=1)
    df['lot_shape_R_A'] = df.apply(lambda x: one_hot(x1['LotShapeA'], 'R-skew'), axis=1)
    df['lot_shape_L_A'] = df.apply(lambda x: one_hot(x1['LotShapeA'], 'L-skew'), axis=1)
    df['lot_shape__B'] = df.apply(lambda x: one_hot(x1['LotShapeB'], '-'), axis=1)
    df['lot_shape_symm_B'] = df.apply(lambda x: one_hot(x1['LotShapeB'], 'Symm'), axis=1)
    df['lot_shape_R_B'] = df.apply(lambda x: one_hot(x1['LotShapeB'], 'R-skew'), axis=1)
    df['lot_shape_L_B'] = df.apply(lambda x: one_hot(x1['LotShapeB'], 'L-skew'), axis=1)
    if only_shapes:
        return df.drop(columns=['LotShapeA', 'LotShapeB'])
    df['R'] = df.apply(lambda x: one_hot(x1['Button'], 'R'), axis=1)
    df['Technion'] = df.apply(lambda x: one_hot(x1['Location'], 'Technion'), axis=1)
    df['M'] = df.apply(lambda x: one_hot(x1['Gender'], 'M'), axis=1)
    return df.drop(columns=['Button', 'Location', 'LotShapeA', 'LotShapeB', 'Gender'])


def feature_extration_individuals():
    df = pd.read_csv('All_estimation_raw_data.csv')
    df = df.drop(columns=['Set', 'Condition', 'RT', 'Feedback', 'block', 'Payoff', 'Forgone'])
    # change from catgorial to one-hot
    df = change_to_one_hot(df)
    # adds opposite games for non amb games
    df = df.append(add_opposite_games(df, is_individuals=True))
    # adds category of num of outcomes
    df['a'] = df.apply(lambda x: Outcomes(x1['LotShapeA'], x1['pHa']), axis=1)
    df['b'] = df.apply(lambda x: Outcomes(x1['LotShapeB'], x1['pHb']), axis=1)
    # adds luck level per game,trial,and participent comnination
    Data = df[
        ['GameID', 'Ha', 'pHa', 'La', 'LotShapeA', 'LotNumA', 'Hb', 'pHb', 'Lb', 'LotShapeB', 'LotNumB', 'Amb', 'Corr',
         'Apay', 'Bpay']].drop_duplicates()
    nProblems = Data.shape[0]
    Data.index = range(nProblems)
    luck_levels_A = []
    luck_levels_B = []
    for prob in range(nProblems):
        Ha = Data['Ha'][prob]
        pHa = Data['pHa'][prob]
        La = Data['La'][prob]
        LotShapeA = lot_shape_convert2(
            Data[['lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A']].values[prob])
        LotNumA = Data['LotNumA'][prob]
        Hb = Data['Hb'][prob]
        pHb = Data['pHb'][prob]
        Lb = Data['Lb'][prob]
        LotShapeB = lot_shape_convert2(
            Data[['lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B']].values[prob])
        LotNumB = Data['LotNumB'][prob]
        Amb = Data['Amb'][prob]
        Corr = Data['Corr'][prob]
        Apay = Data['Apay'][prob]
        Bpay = Data['Bpay'][prob]
        luck_levels_A.append(luck_level(Ha, pHa, La, LotShapeA, LotNumA, Apay))
        luck_levels_B.append(luck_level(Hb, pHb, Lb, LotShapeB, LotNumB, Bpay))
    Data['luck_level_A'] = luck_levels_A
    Data['luck_level_B'] = luck_levels_B
    df = df.merge(Data[['GameID', 'luck_level_A', 'luck_level_B', 'Apay', 'Bpay']], on=['GameID', 'Apay', 'Bpay'])
    # adds biases choises per game and trial
    unbiased, uniform, pessimism, sign, GameID, Trial = [], [], [], [], [], []
    Data = df[['GameID', 'Ha', 'pHa', 'La', 'LotShapeA', 'LotNumA', 'Hb', 'pHb', 'Lb', 'LotShapeB', 'LotNumB', 'Amb',
               'Corr']].drop_duplicates()
    nProblems = Data.shape[0]
    Data.index = range(nProblems)
    for prob in range(nProblems):
        Ha = Data['Ha'][prob]
        pHa = Data['pHa'][prob]
        La = Data['La'][prob]
        LotShapeA = lot_shape_convert2(
            Data[['lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A']].values[prob])
        LotNumA = Data['LotNumA'][prob]
        Hb = Data['Hb'][prob]
        pHb = Data['pHb'][prob]
        Lb = Data['Lb'][prob]
        LotShapeB = lot_shape_convert2(
            Data[['lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B']].values[prob])
        LotNumB = Data['LotNumB'][prob]
        Amb = Data['Amb'][prob]
        Corr = Data['Corr'][prob]
        simPred_unbiased, simPred_uniform, simPred_pessimism, simPred_sign = CPC15_BEASTsimulation_biased(Ha, pHa, La,
                                                                                                          LotShapeA,
                                                                                                          LotNumA, Hb,
                                                                                                          pHb, Lb,
                                                                                                          LotShapeB,
                                                                                                          LotNumB, Amb,
                                                                                                          Corr)
        unbiased.append(simPred_unbiased)
        uniform.append(simPred_uniform)
        pessimism.append(simPred_pessimism)
        sign.append(simPred_sign)
        GameID.append([Data['GameID'][prob] for i in range(25)])
        Trial.append([i + 1 for i in range(25)])
    unbiased_list = []
    uniform_list = []
    pessimism_list = []
    sign_list = []
    GameID_list = []
    Trial_list = []
    for i in range(len(unbiased)):
        for j in range(25):
            unbiased_list.append(unbiased[i][j][0])
            uniform_list.append(uniform[i][j][0])
            pessimism_list.append(pessimism[i][j][0])
            sign_list.append(sign[i][j][0])
            GameID_list.append(GameID[i][j])
            Trial_list.append(Trial[i][j])
    tmp = pd.DataFrame.from_dict(
        {'Unbiased': unbiased_list, 'Uniform': uniform_list, 'Pessimism': pessimism_list, 'Sign': sign_list,
         'GameID': GameID_list, 'Trial': Trial_list})
    grouped = tmp.groupby(['GameID']).agg(
        {'Sign': lambda x: list(x), 'Unbiased': lambda x: list(x), 'Uniform': lambda x: list(x),
         'Pessimism': lambda x: list(x)})
    for i in range(25):
        for column in ['Unbiased', 'Uniform', 'Pessimism', 'Sign']:
            grouped[column + '_' + str(i + 1)] = grouped.apply(lambda x: value_trial(x1[column], i), axis=1)
    for column in ['Unbiased', 'Uniform', 'Pessimism', 'Sign']:
        grouped[column + '_Mean'] = grouped[column].apply(lambda x: np.mean(np.array(x)))
        grouped[column + '_Var'] = grouped[column].apply(lambda x: np.var(np.array(x)))
    grouped = grouped.drop(columns=['Unbiased', 'Uniform', 'Pessimism', 'Sign'])
    df = grouped.merge(df, on=['GameID'])
    # adds psychological features
    df2 = df[df['GameID'] < 211].merge(pd.read_csv('RealData270.csv')[
                                           ['GameID', 'diffEV', 'diffSDs', 'diffMins', 'diffMaxs', 'diffUV', 'RatioMin',
                                            'SignMax', 'pBbet_Unbiased1', 'pBbet_UnbiasedFB', 'pBbet_Uniform',
                                            'pBbet_Sign1', 'pBbet_SignFB', 'Dom', 'diffBEV0', 'diffBEVfb',
                                            'diffSignEV']], on='GameID').sort_values(
        by=['SubjID', 'Order', 'GameID', 'Trial'])
    # df = df2.append(add_psychological_features(df[df['GameID'] > 210][['GameID', 'Ha', 'pHa', 'La', 'lot_shape__A','lot_shape_symm_A','lot_shape_R_A','lot_shape_L_A', 'LotNumA', 'Hb', 'pHb', 'Lb',  'lot_shape__B','lot_shape_symm_B','lot_shape_R_B','lot_shape_L_B',  'LotNumB', 'Amb','Corr']].drop_duplicates()))
    # adds weather B is good
    df['Good'] = df.apply(lambda x: is_good(x1['Bpay'], x1['Apay']), axis=1)
    final = df[df['Trial'] < 6].drop(columns=['luck_level_A', 'luck_level_B'])
    for i in range(-50, 257):
        for val in ['Apay', 'Bpay']:
            df[val + '_' + str(i)] = 0
    for trial2 in range(6, 26):
        df['Good_' + str(trial2)] = 0.5
    for trial in range(6, 26):
        print(trial)
        relevant = df[df['Trial'] == trial]
        length = len(relevant)
        relevant.index = range(length)
        for val in ['Apay', 'Bpay']:
            pays = relevant[val]
            for value in range(-50, 257):
                if trial == 6:
                    tmp = np.zeros(length)
                else:
                    tmp = old[val + '_' + str(value)].values
                for i in range(length):
                    if value == pays[i]:
                        tmp[i] += 1
                relevant[val + '_' + str(value)] = tmp
            if trial == 6:
                relevant[val + '_List'] = relevant.apply(lambda x: [x1[val]], axis=1)
            else:
                relevant[val + '_List'] = old[val + '_List']
                relevant[val + '_List'] = relevant.apply(lambda x: append(x1[val + '_List'], x1[val]), axis=1)
            relevant[val + '_Mean'] = relevant.apply(lambda x: np.mean(x1[val + '_List']), axis=1)
            relevant[val + '_Var'] = relevant.apply(lambda x: np.var(x1[val + '_List']), axis=1)
        for val in ['luck_level_A', 'luck_level_B']:
            if trial == 6:
                relevant[val + '_List'] = relevant.apply(lambda x: [x1[val]], axis=1)
            else:
                relevant[val + '_List'] = old[val + '_List']
                relevant[val + '_List'] = relevant.apply(lambda x: append(x1[val + '_List'], x1[val]), axis=1)
            relevant['luck_level_Mean'] = relevant.apply(lambda x: np.mean(x1[val + '_List']), axis=1)
        for trial2 in range(6, trial + 1):
            if trial == trial2:
                relevant['Good_' + str(trial2)] = relevant['Good']
            else:
                relevant['Good_' + str(trial2)] = old['Good']
        old = relevant
        old.to_csv(str(trial) + '.csv', index=False)
        relevant = relevant.drop(
            columns=['luck_level_A', 'luck_level_B', 'luck_level_A_List', 'luck_level_B_List', 'Apay_List',
                     'Bpay_List'])
        final = final.append(relevant)
    final = final.sort_values(by=['SubjID', 'Order', 'GameID', 'Trial']).drop_duplicates()
    final.to_csv('final.csv', index=False)


def feature_extraction_aggregate():
    df = pd.read_csv('RealData270.csv')
    # df = df.append(add_psychological_features(add_opposite_games(df=df,is_individuals=False,baseline='BEASTpred'),is_individuals=False,baseline='BEASTpred'))
    df['MSE'] = df.apply(lambda x: mean_squared_error(x1[['B.' + str(i) for i in range(1, 6)]],
                                                      x1[['BEASTpred.' + str(i) for i in range(1, 6)]]), axis=1)

    def good_bad(MSE):
        if MSE < 0.01:
            return 1
        else:
            return 0

    df['good_bad'] = df.apply(lambda x: good_bad(x1['MSE']), axis=1)
    df.to_csv('aggregate.csv', index=False)


def feature_extraction_per_game_and_player():
    df = pd.read_csv('All_estimation_raw_data.csv')
    df = change_to_one_hot(df)

    def baseline(SubjID, GameID, Trial,
                 df):  # set as baseline the average of all of the other participents' decision per trial and game
        return df[(df['SubjID'] != SubjID) & (df['GameID'] == GameID) & (df['Trial'] == Trial)]['B'].mean()

    df['baseline'] = df.apply(lambda x: baseline(x1['SubjID'], x1['GameID'], x1['Trial'], df), axis=1)
    print("baseline done")
    per_game_and_player = df.drop(
        columns=['Trial', 'R', 'B', 'Payoff', 'Forgone', 'RT', 'Apay', 'Bpay', 'Feedback', 'block']).drop_duplicates()

    def avg_per_block(SubjID, GameID, block, B,
                      df):  # calculates the average decision/baseline per block and game of each participent
        return df[(df['SubjID'] == SubjID) & (df['GameID'] == GameID) & (df['block'] == block)][B].mean()

    for block in range(1, 6):
        per_game_and_player['B.' + str(block)] = per_game_and_player.apply(
            lambda x: avg_per_block(x1['SubjID'], x1['GameID'], block, 'B', df), axis=1)
        print("B." + str(block) + " done")
        per_game_and_player['B_baseline.' + str(block)] = per_game_and_player.apply(
            lambda x: avg_per_block(x1['SubjID'], x1['GameID'], block, 'baseline', df), axis=1)
        print("B_baseline." + str(block) + " done")
        per_game_and_player['Apay.' + str(block)] = per_game_and_player.apply(
            lambda x: avg_per_block(x1['SubjID'], x1['GameID'], block, 'Apay', df), axis=1)
        print("Apay." + str(block) + " done")
        per_game_and_player['Bpay.' + str(block)] = per_game_and_player.apply(
            lambda x: avg_per_block(x1['SubjID'], x1['GameID'], block, 'Bpay', df), axis=1)
        print("Bpay." + str(block) + " done")
    per_game_and_player['MSE'] = per_game_and_player.apply(
        lambda x: mean_squared_error(x1[['B.' + str(i) for i in range(1, 6)]],
                                     x1[['B_baeline.' + str(i) for i in range(1, 6)]]), axis=1)
    print("MSE done")

    def good_bad(MSE):
        if MSE < 0.1:
            return 1
        else:
            return 0

    per_game_and_player['good_bad'] = per_game_and_player.apply(lambda x: good_bad(x1['MSE']), axis=1)
    # per_game_and_player = per_game_and_player.append(add_psychological_features(add_opposite_games(df=per_game_and_player,is_individuals=False,baseline='B_baseline',per_game_and_player=True),is_individuals=False,baseline='B_baseline',per_game_and_player=True))
    per_game_and_player.to_csv('per_game_and_player.csv', index=False)

    def avg(df, ID, column, val):
        return df[df[column] == ID][val].mean()

    def var(df, ID, column, val):
        return df[df[column] == ID][val].var()

    per_player = per_game_and_player[['M', 'Age', 'Technion', 'SubjID']].drop_duplicates()
    per_player['MSE'] = per_player.apply(lambda x: avg(df, x1['SubjID']), axis=1)
    per_player['MSE_var'] = per_player.apply(lambda x: var(df, x1['SubjID']), axis=1)
    per_player['good_bad'] = per_player.apply(lambda x: good_bad(x1['MSE']), axis=1)
    per_player.to_csv('per_player.csv', index=False)
    per_game = per_game_and_player[per_game_and_player.drop(
        columns=['M', 'Age', 'Technion', 'SubjID', 'MSE', 'Apay.1', 'Bpay.1', 'B.1', 'BEASTpred.1', 'Apay.2', 'Bpay.2',
                 'B.2', 'BEASTpred.2', 'Apay.3', 'Bpay.3', 'B.3', 'BEASTpred.3', 'Apay.4', 'Bpay.4', 'B.4',
                 'BEASTpred.4', 'Apay.5', 'Bpay.5', 'B.5', 'BEASTpred.5']).columns].drop_duplicates()
    per_game['MSE'] = per_game.apply(lambda x: avg(df, x1['SubjID'], 'MSE'), axis=1)
    per_game['MSE_var'] = per_game.apply(lambda x: var(df, x1['SubjID'], 'MSE'), axis=1)
    per_game['good_bad'] = per_game.apply(lambda x: good_bad(x1['MSE']), axis=1)
    per_game['Apay'] = per_game.apply(lambda x: avg(df, x1['SubjID'], 'Apay'), axis=1)
    per_game['Bpay'] = per_game.apply(lambda x: avg(df, x1['SubjID'], 'Bpay'), axis=1)
    per_game['Apay_var'] = per_game.apply(lambda x: var(df, x1['SubjID'], 'Apay'), axis=1)
    per_game['Bpay_var'] = per_game.apply(lambda x: var(df, x1['SubjID'], 'Bpay'), axis=1)
    per_game.to_csv('per_game.csv', index=False)


def to_B1(train):
    df2 = train[train['block'] == 1].drop(columns=['BEASTpred', 'B', 'block', 'Feedback'])
    df2['B.1'] = list(train[train['block'] == 1]['B'])
    df2['B.2'] = list(train[train['block'] == 2]['B'])
    df2['B.3'] = list(train[train['block'] == 3]['B'])
    df2['B.4'] = list(train[train['block'] == 4]['B'])
    df2['B.5'] = list(train[train['block'] == 5]['B'])
    df2['BEASTpred.1'] = list(train[train['block'] == 1]['BEASTpred'])
    df2['BEASTpred.2'] = list(train[train['block'] == 2]['BEASTpred'])
    df2['BEASTpred.3'] = list(train[train['block'] == 3]['BEASTpred'])
    df2['BEASTpred.4'] = list(train[train['block'] == 4]['BEASTpred'])
    df2['BEASTpred.5'] = list(train[train['block'] == 5]['BEASTpred'])
    return df2


def logistic(B, X):
    return 1 / (1 + np.exp(-B * X))


def calc_bias_weights(B, BEVb, BEVa, val_dist, probs,probs0, Corr):
    kapa = {}
    for k1 in ['unb', 'uni', 'pes', 'sig']:
        kapa[k1] = [0, 0]
        for k2 in ['unb', 'uni', 'pes', 'sig']:
            kapa[k1 + '_' + k2] = [0, 0]
            for k3 in ['unb', 'uni', 'pes', 'sig']:
                kapa[k1 + '_' + k2 + '_' + k3] = [0, 0]
    for k1 in ['unb', 'uni', 'pes', 'sig']:
        val_A1, dist_A1, val_B1, dist_B1 = val_dist[k1][0], val_dist[k1][1], val_dist[k1][2], val_dist[k1][3]
        for i1 in range(val_A1.shape[0]):
            for j1 in range(val_B1.shape[0]):
                if min(dist_A1[i1 + 1], dist_B1[j1 + 1]) > max(dist_A1[i1], dist_B1[j1]):
                    kapa[k1][0] += (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1],dist_B1[j1])) * logistic(B,BEVb - BEVa +val_B1[j1] -val_A1[i1])
                    for k2 in ['unb', 'uni', 'pes', 'sig']:
                        val_A2, dist_A2, val_B2, dist_B2 = val_dist[k2][0], val_dist[k2][1], val_dist[k2][2],val_dist[k2][3]
                        for i2 in range(val_A2.shape[0]):
                            for j2 in range(val_B2.shape[0]):
                                if min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]):
                                    kapa[k1 + '_' + k2][0] += (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                    for k3 in ['unb', 'uni', 'pes', 'sig']:
                                        val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                        for i3 in range(val_A3.shape[0]):
                                            for j3 in range(val_B3.shape[0]):
                                                if min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]):
                                                    kapa[k1 + '_' + k2 + '_' + k3][0] += (min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3], dist_B3[j3])) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                if Corr == -1:
                    if min(dist_A1[i1 + 1], 1 - dist_B1[j1]) > max(dist_A1[i1], 1 - dist_B1[j1 + 1]) and k1 in ('unb', 'sig'):
                        kapa[k1][1] += (min(dist_A1[i1 + 1], 1 - dist_B1[j1]) - max(dist_A1[i1],1 - dist_B1[j1 + 1])) * logistic(B, BEVb - BEVa +val_B1[j1] -val_A1[i1])
                        for k2 in ['unb', 'uni', 'pes', 'sig']:
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[k2][0], val_dist[k2][1], val_dist[k2][2],val_dist[k2][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if min(dist_A2[i2 + 1], 1 - dist_B2[j2]) > max(dist_A2[i2],1 - dist_B2[j2 + 1]) and k2 in ('unb', 'sig'):
                                        kapa[k1 + '_' + k2][1] += (min(dist_A1[i1 + 1], 1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1],1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],1 - dist_B3[j3 + 1]) and k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1], 1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])) * (min(dist_A3[i3 + 1],1 - dist_B3[j3]) - max(dist_A3[i3], 1 - dist_B3[j3 + 1])) * logistic(B, BEVb - BEVa + (val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1],1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])) * (min(dist_A3[i3 + 1], dist_B3[j3 + 1]) - max(dist_A3[i3],dist_B3[j3])) * logistic(B,BEVb - BEVa + (val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and k2 in ('uni', 'pes'):
                                        kapa[k1 + '_' + k2][1] += (min(dist_A1[i1 + 1], 1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],1 - dist_B3[j3 + 1]) and k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2],dist_B2[j2])) * (min(dist_A3[i3 + 1],1 -dist_B3[j3]) - max(dist_A3[i3],1 - dist_B3[j3 + 1])) * logistic(B,BEVb - BEVa + (val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2],dist_B2[j2])) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3],dist_B3[j3])) * logistic(B,BEVb - BEVa + (val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                    elif min(dist_A1[i1 + 1], dist_B1[j1 + 1]) > max(dist_A1[i1], dist_B1[j1]) and k1 in ('uni', 'pes'):
                        kapa[k1][1] += (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1],dist_B1[j1])) * logistic(B,BEVb - BEVa +val_B1[j1] -val_A1[i1])
                        for k2 in ['unb', 'uni', 'pes', 'sig']:
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[k2][0], val_dist[k2][1], val_dist[k2][2],val_dist[k2][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if min(dist_A2[i2 + 1], 1 - dist_B2[j2]) > max(dist_A2[i2],1 - dist_B2[j2 + 1]) and k2 in ('unb', 'sig'):
                                        kapa[k1 + '_' + k2][1] += (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],1 - dist_B3[j3 + 1]) and k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])) * (min(dist_A3[i3 + 1],1 - dist_B3[j3]) - max(dist_A3[i3], 1 - dist_B3[j3 + 1])) * logistic(B,BEVb - BEVa + (val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3], dist_B3[j3])) * logistic(B,BEVb - BEVa + (val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[ i1] + val_A2[i2] +val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and k2 in ('uni', 'pes'):
                                        kapa[k1 + '_' + k2][1] += (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],1 - dist_B3[j3 + 1]) and k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (min(dist_A3[i3 + 1],1 - dist_B3[j3]) - max(dist_A3[i3], 1 - dist_B3[j3 + 1])) * logistic(B,BEVb - BEVa + (val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] + val_A3[ i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3], dist_B3[j3])) * logistic(B,BEVb - BEVa + (val_B1[j1] + val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)

                elif Corr == 0:
                    if k1 in ('unb', 'sig'):
                        kapa[k1][1] += (dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] - dist_B1[j1]) * logistic(B,BEVb - BEVa +val_B1[j1] -val_A1[i1])
                        for k2 in ['unb', 'uni', 'pes', 'sig']:
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[k2][0], val_dist[k2][1], val_dist[k2][2],val_dist[k2][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if k2 in ('unb', 'sig'):
                                        kapa[k1 + '_' + k2][1] += (dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] - dist_B1[j1]) * (dist_A2[i2 + 1] - dist_A2[i2]) * (dist_B2[j2 + 1] - dist_B2[j2]) * logistic(B,BEVb - BEVa + (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] -dist_B1[j1]) * (dist_A2[i2 + 1] -dist_A2[i2]) * (dist_B2[j2 + 1] - dist_B2[j2]) * (dist_A3[i3 + 1] - dist_A3[i3]) * (dist_B3[j3 + 1] -dist_B3[j3]) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] -dist_B1[j1]) * (dist_A2[i2 + 1] -dist_A2[i2]) * (dist_B2[j2 + 1] -dist_B2[j2]) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3],dist_B3[j3])) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and k2 in ('uni', 'pes'):
                                        kapa[k1 + '_' + k2][1] += (dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] - dist_B1[j1]) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * logistic(B,BEVb - BEVa + (val_B1[j1] +val_B2[j2]) / 2 - (val_A1[i1] +val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] -dist_B1[j1]) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2],dist_B2[j2])) * (dist_A3[i3 + 1] -dist_A3[i3]) * (dist_B3[j3 + 1] -dist_B3[j3]) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] -dist_B1[j1]) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (min(dist_A3[i3 + 1], dist_B3[j3 + 1]) - max(dist_A3[i3],dist_B3[j3])) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                    elif min(dist_A1[i1 + 1], dist_B1[j1 + 1]) > max(dist_A1[i1], dist_B1[j1]) and k1 in ('uni', 'pes'):
                        kapa[k1][1] += (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1],dist_B1[j1])) * logistic(B,BEVb - BEVa +val_B1[j1] -val_A1[i1])
                        for k2 in ['unb', 'uni', 'pes', 'sig']:
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[k2][0], val_dist[k2][1], val_dist[k2][2],val_dist[k2][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if k2 in ('unb', 'sig'):
                                        kapa[k1 + '_' + k2][1] += (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (dist_A2[i2 + 1] - dist_A2[i2]) * (dist_B2[j2 + 1] - dist_B2[j2]) * logistic(B,BEVb - BEVa + (val_B1[j1] +val_B2[j2]) / 2 - (val_A1[i1] +val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (dist_A2[i2 + 1] - dist_A2[i2]) * (dist_B2[j2 + 1] - dist_B2[j2]) * (dist_A3[i3 + 1] -dist_A3[i3]) * (dist_B3[j3 + 1] -dist_B3[j3]) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (dist_A2[i2 + 1] - dist_A2[i2]) * (dist_B2[j2 + 1] - dist_B2[j2]) * (min(dist_A3[i3 + 1], dist_B3[j3 + 1]) - max(dist_A3[i3],dist_B3[j3])) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and k2 in ('uni', 'pes'):
                                        kapa[k1 + '_' + k2][1] += (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * logistic(B,BEVb - BEVa + (val_B1[j1] +val_B2[j2]) / 2 - (val_A1[i1] +val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (dist_A3[i3 + 1] - dist_A3[i3]) * (dist_B3[j3 + 1] - dist_B3[j3]) * logistic(B,BEVb - BEVa + (val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1] += (min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3], dist_B3[j3])) * logistic(B, BEVb - BEVa + (val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)

    for k1 in ['unb', 'uni', 'pes', 'sig']:
        probs0[k1].append(kapa[k1][0] / 3)
        for k2 in ['unb', 'uni', 'pes', 'sig']:
            probs0[k1 + '_' + k2].append(kapa[k1 + '_' + k2][0] / 3)
            for k3 in ['unb', 'uni', 'pes', 'sig']:
                probs0[k1 + '_' + k2 + '_' + k3].append(kapa[k1 + '_' + k2 + '_' + k3][0] / 3)
    if Corr == 1:
        for k1 in ['unb', 'uni', 'pes', 'sig']:
            probs[k1].append(kapa[k1][0] / 3)
            for k2 in ['unb', 'uni', 'pes', 'sig']:
                probs[k1 + '_' + k2].append(kapa[k1 + '_' + k2][0] / 3)
                for k3 in ['unb', 'uni', 'pes', 'sig']:
                    probs[k1 + '_' + k2 + '_' + k3].append(kapa[k1 + '_' + k2 + '_' + k3][0] / 3)
    else:
        for k1 in ['unb', 'uni', 'pes', 'sig']:
            probs[k1].append(kapa[k1][1] / 3)
            for k2 in ['unb', 'uni', 'pes', 'sig']:
                probs[k1 + '_' + k2].append(kapa[k1 + '_' + k2][1] / 3)
                for k3 in ['unb', 'uni', 'pes', 'sig']:
                    probs[k1 + '_' + k2 + '_' + k3].append(kapa[k1 + '_' + k2 + '_' + k3][1] / 3)
    return probs,probs0


def sigmoid_weights(val_dist, Corr, probs, differences, probs0, differences0):
    kapa, st_difference = {}, {}
    for k1 in ['unb', 'uni', 'pes', 'sig']:
        kapa[k1] = [],[]
        st_difference[k1] = [],[]
        for k2 in ['unb', 'uni', 'pes', 'sig']:
            kapa[k1 + '_' + k2] = [],[]
            st_difference[k1 + '_' + k2] = [],[]
            for k3 in ['unb', 'uni', 'pes', 'sig']:
                kapa[k1 + '_' + k2 + '_' + k3] = [],[]
                st_difference[k1 + '_' + k2 + '_' + k3] = [],[]
    for k1 in ['unb', 'uni', 'pes', 'sig']:
        val_A1, dist_A1, val_B1, dist_B1 = val_dist[k1][0], val_dist[k1][1], val_dist[k1][2], val_dist[k1][3]
        for i1 in range(val_A1.shape[0]):
            for j1 in range(val_B1.shape[0]):
                if min(dist_A1[i1 + 1], dist_B1[j1 + 1]) > max(dist_A1[i1], dist_B1[j1]):
                    kapa[k1][0].append(min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1],dist_B1[j1]))
                    st_difference[k1][0].append(val_B1[j1] -val_A1[i1])
                    for k2 in ['unb', 'uni', 'pes', 'sig']:
                        val_A2, dist_A2, val_B2, dist_B2 = val_dist[k2][0], val_dist[k2][1], val_dist[k2][2],val_dist[k2][3]
                        for i2 in range(val_A2.shape[0]):
                            for j2 in range(val_B2.shape[0]):
                                if min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]):
                                    kapa[k1 + '_' + k2][0].append((min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])))
                                    st_difference[k1 + '_' + k2][0].append((val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                    for k3 in ['unb', 'uni', 'pes', 'sig']:
                                        val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                        for i3 in range(val_A3.shape[0]):
                                            for j3 in range(val_B3.shape[0]):
                                                if min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]):
                                                    kapa[k1 + '_' + k2 + '_' + k3][0].append((min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3], dist_B3[j3])))
                                                    st_difference[k1 + '_' + k2 + '_' + k3][0].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                if Corr == -1:
                    if min(dist_A1[i1 + 1], 1 - dist_B1[j1]) > max(dist_A1[i1], 1 - dist_B1[j1 + 1]) and k1 in ('unb', 'sig'):
                        kapa[k1][1].append(min(dist_A1[i1 + 1], 1 - dist_B1[j1]) - max(dist_A1[i1],1 - dist_B1[j1 + 1]))
                        st_difference[k1][1].append(val_B1[j1] -val_A1[i1])
                        for k2 in ['unb', 'uni', 'pes', 'sig']:
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[k2][0], val_dist[k2][1], val_dist[k2][2],val_dist[k2][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if min(dist_A2[i2 + 1], 1 - dist_B2[j2]) > max(dist_A2[i2],1 - dist_B2[j2 + 1]) and k2 in ('unb', 'sig'):
                                        kapa[k1 + '_' + k2][1].append((min(dist_A1[i1 + 1], 1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1],1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])))
                                        st_difference[k1 + '_' + k2][1].append((val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],1 - dist_B3[j3 + 1]) and k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1], 1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])) * (min(dist_A3[i3 + 1],1 - dist_B3[j3]) - max(dist_A3[i3], 1 - dist_B3[j3 + 1])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1],1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])) * (min(dist_A3[i3 + 1], dist_B3[j3 + 1]) - max(dist_A3[i3],dist_B3[j3])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and k2 in ('uni', 'pes'):
                                        kapa[k1 + '_' + k2][1].append((min(dist_A1[i1 + 1], 1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])))
                                        st_difference[k1 + '_' + k2][1].append((val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],1 - dist_B3[j3 + 1]) and k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2],dist_B2[j2])) * (min(dist_A3[i3 + 1],1 -dist_B3[j3]) - max(dist_A3[i3],1 - dist_B3[j3 + 1])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2],dist_B2[j2])) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3],dist_B3[j3])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                    elif min(dist_A1[i1 + 1], dist_B1[j1 + 1]) > max(dist_A1[i1], dist_B1[j1]) and k1 in ('uni', 'pes'):
                        kapa[k1][1].append(min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1],dist_B1[j1]))
                        st_difference[k1][1].append(val_B1[j1] -val_A1[i1])
                        for k2 in ['unb', 'uni', 'pes', 'sig']:
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[k2][0], val_dist[k2][1], val_dist[k2][2],val_dist[k2][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if min(dist_A2[i2 + 1], 1 - dist_B2[j2]) > max(dist_A2[i2],1 - dist_B2[j2 + 1]) and k2 in ('unb', 'sig'):
                                        kapa[k1 + '_' + k2][1].append((min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])))
                                        st_difference[k1 + '_' + k2][1].append((val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],1 - dist_B3[j3 + 1]) and k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])) * (min(dist_A3[i3 + 1],1 - dist_B3[j3]) - max(dist_A3[i3], 1 - dist_B3[j3 + 1])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],1 - dist_B2[j2]) - max(dist_A2[i2], 1 - dist_B2[j2 + 1])) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3], dist_B3[j3])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[ i1] + val_A2[i2] +val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and k2 in ('uni', 'pes'):
                                        kapa[k1 + '_' + k2][1].append((min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])))
                                        st_difference[k1 + '_' + k2][1].append((val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],1 - dist_B3[j3 + 1]) and k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (min(dist_A3[i3 + 1],1 - dist_B3[j3]) - max(dist_A3[i3], 1 - dist_B3[j3 + 1])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] + val_A3[ i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3], dist_B3[j3])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] + val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                elif Corr == 0:
                    if k1 in ('unb', 'sig'):
                        kapa[k1][1].append((dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] - dist_B1[j1]))
                        st_difference[k1][1].append(val_B1[j1] -val_A1[i1])
                        for k2 in ['unb', 'uni', 'pes', 'sig']:
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[k2][0], val_dist[k2][1], val_dist[k2][2],val_dist[k2][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if k2 in ('unb', 'sig'):
                                        kapa[k1 + '_' + k2][1].append((dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] - dist_B1[j1]) * (dist_A2[i2 + 1] - dist_A2[i2]) * (dist_B2[j2 + 1] - dist_B2[j2]))
                                        st_difference[k1 + '_' + k2][1].append((val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] -dist_B1[j1]) * (dist_A2[i2 + 1] -dist_A2[i2]) * (dist_B2[j2 + 1] - dist_B2[j2]) * (dist_A3[i3 + 1] - dist_A3[i3]) * (dist_B3[j3 + 1] -dist_B3[j3]))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] -dist_B1[j1]) * (dist_A2[i2 + 1] -dist_A2[i2]) * (dist_B2[j2 + 1] -dist_B2[j2]) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3],dist_B3[j3])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and k2 in ('uni', 'pes'):
                                        kapa[k1 + '_' + k2][1].append((dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] - dist_B1[j1]) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])))
                                        st_difference[k1 + '_' + k2][1].append((val_B1[j1] +val_B2[j2]) / 2 - (val_A1[i1] +val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] -dist_B1[j1]) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2],dist_B2[j2])) * (dist_A3[i3 + 1] -dist_A3[i3]) * (dist_B3[j3 + 1] -dist_B3[j3]))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] -dist_B1[j1]) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (min(dist_A3[i3 + 1], dist_B3[j3 + 1]) - max(dist_A3[i3],dist_B3[j3])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                    elif min(dist_A1[i1 + 1], dist_B1[j1 + 1]) > max(dist_A1[i1], dist_B1[j1]) and k1 in ('uni', 'pes'):
                        kapa[k1][1].append(min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1],dist_B1[j1]))
                        st_difference[k1][1].append(val_B1[j1] -val_A1[i1])
                        for k2 in ['unb', 'uni', 'pes', 'sig']:
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[k2][0], val_dist[k2][1], val_dist[k2][2],val_dist[k2][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if k2 in ('unb', 'sig'):
                                        kapa[k1 + '_' + k2][1].append((min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (dist_A2[i2 + 1] - dist_A2[i2]) * (dist_B2[j2 + 1] - dist_B2[j2]))
                                        st_difference[k1 + '_' + k2][1].append((val_B1[j1] +val_B2[j2]) / 2 - (val_A1[i1] +val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (dist_A2[i2 + 1] - dist_A2[i2]) * (dist_B2[j2 + 1] - dist_B2[j2]) * (dist_A3[i3 + 1] -dist_A3[i3]) * (dist_B3[j3 + 1] -dist_B3[j3]))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (dist_A2[i2 + 1] - dist_A2[i2]) * (dist_B2[j2 + 1] - dist_B2[j2]) * (min(dist_A3[i3 + 1], dist_B3[j3 + 1]) - max(dist_A3[i3],dist_B3[j3])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and k2 in ('uni', 'pes'):
                                        kapa[k1 + '_' + k2][1].append((min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])))
                                        st_difference[k1 + '_' + k2][1].append((val_B1[j1] +val_B2[j2]) / 2 - (val_A1[i1] +val_A2[i2]) / 2)
                                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[k3][0], val_dist[k3][1],val_dist[k3][2], val_dist[k3][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if k3 in ('unb', 'sig'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (dist_A3[i3 + 1] - dist_A3[i3]) * (dist_B3[j3 + 1] - dist_B3[j3]))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] +val_B2[j2] +val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],dist_B3[j3]) and k3 in ('uni', 'pes'):
                                                        kapa[k1 + '_' + k2 + '_' + k3][1].append((min(dist_A1[i1 + 1],dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (min(dist_A2[i2 + 1],dist_B2[j2 + 1]) - max(dist_A2[i2], dist_B2[j2])) * (min(dist_A3[i3 + 1],dist_B3[j3 + 1]) - max(dist_A3[i3], dist_B3[j3])))
                                                        st_difference[k1 + '_' + k2 + '_' + k3][1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (val_A1[i1] +val_A2[i2] +val_A3[i3]) / 3)

    for k1 in ['unb', 'uni', 'pes', 'sig']:
        probs0[k1].append(np.array(kapa[k1][0]) / 3)
        differences0[k1].append(np.array(st_difference[k1][0]))
        for k2 in ['unb', 'uni', 'pes', 'sig']:
            probs0[k1 + '_' + k2].append(np.array(kapa[k1 + '_' + k2][0]) / 3)
            differences0[k1 + '_' + k2].append(np.array(st_difference[k1 + '_' + k2][0]))
            for k3 in ['unb', 'uni', 'pes', 'sig']:
                probs0[k1 + '_' + k2 + '_' + k3].append(np.array(kapa[k1 + '_' + k2 + '_' + k3][0]) / 3)
                differences0[k1 + '_' + k2 + '_' + k3].append(np.array(st_difference[k1 + '_' + k2 + '_' + k3][0]))
    if Corr == 1:
        for k1 in ['unb', 'uni', 'pes', 'sig']:
            probs[k1].append(np.array(kapa[k1][0]) / 3)
            differences[k1].append(np.array(st_difference[k1][0]))
            for k2 in ['unb', 'uni', 'pes', 'sig']:
                probs[k1 + '_' + k2].append(np.array(kapa[k1 + '_' + k2][0]) / 3)
                differences[k1 + '_' + k2].append(np.array(st_difference[k1 + '_' + k2][0]))
                for k3 in ['unb', 'uni', 'pes', 'sig']:
                    probs[k1 + '_' + k2 + '_' + k3].append(np.array(kapa[k1 + '_' + k2 + '_' + k3][0]) / 3)
                    differences[k1 + '_' + k2 + '_' + k3].append(np.array(st_difference[k1 + '_' + k2 + '_' + k3][0]))
    else:
        for k1 in ['unb', 'uni', 'pes', 'sig']:
            probs[k1].append(np.array(kapa[k1][1]) / 3)
            differences[k1].append(np.array(st_difference[k1][1]))
            for k2 in ['unb', 'uni', 'pes', 'sig']:
                probs[k1 + '_' + k2].append(np.array(kapa[k1 + '_' + k2][1]) / 3)
                differences[k1 + '_' + k2].append(np.array(st_difference[k1 + '_' + k2][1]))
                for k3 in ['unb', 'uni', 'pes', 'sig']:
                    probs[k1 + '_' + k2 + '_' + k3].append(np.array(kapa[k1 + '_' + k2 + '_' + k3][1]) / 3)
                    differences[k1 + '_' + k2 + '_' + k3].append(np.array(st_difference[k1 + '_' + k2 + '_' + k3][1]))
    return probs, differences, probs0, differences0


def add_bias_weights(Data, name, B=None, sigmoid=True):
    BEVas, BEVbs = [], []
    probs, differences, probs0, differences0 = {}, {}, {}, {}
    for k1 in ['unb', 'uni', 'pes', 'sig']:
        probs[k1], differences[k1], probs0[k1], differences0[k1] = [], [], [], []
        for k2 in ['unb', 'uni', 'pes', 'sig']:
            probs[k1 + '_' + k2], differences[k1 + '_' + k2], probs0[k1 + '_' + k2], differences0[
                k1 + '_' + k2] = [], [], [], []
            for k3 in ['unb', 'uni', 'pes', 'sig']:
                probs[k1 + '_' + k2 + '_' + k3], differences[k1 + '_' + k2 + '_' + k3], probs0[
                    k1 + '_' + k2 + '_' + k3], differences0[k1 + '_' + k2 + '_' + k3] = [], [], [], []
    nProblems = Data.shape[0]
    Data.index = range(nProblems)
    for prob in range(nProblems):
        print(prob)
        Ha = Data['Ha'][prob]
        pHa = Data['pHa'][prob]
        La = Data['La'][prob]
        LotShapeA = Data['LotShapeA'][prob]
        LotNumA = int(Data['LotNumA'][prob])
        Hb = Data['Hb'][prob]
        pHb = Data['pHb'][prob]
        Lb = Data['Lb'][prob]
        LotShapeB = Data['LotShapeB'][prob]
        LotNumB = int(Data['LotNumB'][prob])
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
        BEVas.append(BEVa)
        BEVbs.append(BEVb)
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
        if sigmoid:
            probs, differences, probs0, differences0 = sigmoid_weights(val_dist, Corr, probs, differences, probs0,
                                                                       differences0)
        else:
            probs,probs0 = calc_bias_weights(B, BEVb, BEVa, val_dist, probs,probs0, Corr)
        if (prob+1)%100==0 or prob==nProblems-1:
            if sigmoid:
                if (prob+1)%100==0:
                    tmp=Data[prob-99:prob+1]
                elif prob==nProblems-1:
                    tmp=Data[prob-prob%100:]
                tmp.index=range(len(tmp))
                tmp['BEVa'] = BEVas
                tmp['BEVb'] = BEVbs
                tmp = tmp[['Problem','BEVa','BEVb','pBias0','pBias1','pBias2','pBias3','pBias4']]
                for key in probs.keys():
                    tmp['Weight_' + key] = probs[key]
                for key in differences.keys():
                    tmp['ST_' + key] = differences[key]
                for key in probs0.keys():
                    tmp['Weight0_' + key] = probs0[key]
                for key in differences0.keys():
                    tmp['ST0_' + key] = differences0[key]
                tmp.to_pickle(name+'_'+str(prob)+'.pkl')
                BEVas, BEVbs = [], []
                probs, differences, probs0, differences0 = {}, {}, {}, {}
                for k1 in ['unb', 'uni', 'pes', 'sig']:
                    probs[k1], differences[k1], probs0[k1], differences0[k1] = [], [], [], []
                    for k2 in ['unb', 'uni', 'pes', 'sig']:
                        probs[k1 + '_' + k2], differences[k1 + '_' + k2], probs0[k1 + '_' + k2], differences0[
                            k1 + '_' + k2] = [], [], [], []
                        for k3 in ['unb', 'uni', 'pes', 'sig']:
                            probs[k1 + '_' + k2 + '_' + k3], differences[k1 + '_' + k2 + '_' + k3], probs0[
                                k1 + '_' + k2 + '_' + k3], differences0[k1 + '_' + k2 + '_' + k3] = [], [], [], []
    if not sigmoid:
        for key in probs.keys():
            Data['Weight_' + key] = probs[key]
        for key in probs0.keys():
            Data['Weight0_' + key] = probs0[key]
        Data.to_csv(name+'.csv', index=False)


def MSE_biases(x,df, label):
    x1=x[0:4]
    x2=x[4:8]
    x3=x[8:12]
    x4=x[12:16]
    x5=x[16:20]
    return ((
                              x1[0] * df['Weight0_unb'] +
                              x1[0] * x1[0] * df['Weight0_unb_unb'] +
                              x1[0] * x1[0] * x1[0] * df['Weight0_unb_unb_unb'] +
                              x1[0] * x1[0] * x1[1] * df['Weight0_unb_unb_uni'] +
                              x1[0] * x1[0] * x1[2] * df['Weight0_unb_unb_pes'] +
                              x1[0] * x1[0] * x1[3] * df['Weight0_unb_unb_sig'] +
                              x1[0] * x1[1] * df['Weight0_unb_uni'] +
                              x1[0] * x1[1] * x1[0] * df['Weight0_unb_uni_unb'] +
                              x1[0] * x1[1] * x1[1] * df['Weight0_unb_uni_uni'] +
                              x1[0] * x1[1] * x1[2] * df['Weight0_unb_uni_pes'] +
                              x1[0] * x1[1] * x1[3] * df['Weight0_unb_uni_sig'] +
                              x1[0] * x1[2] * df['Weight0_unb_pes'] +
                              x1[0] * x1[2] * x1[0] * df['Weight0_unb_pes_unb'] +
                              x1[0] * x1[2] * x1[1] * df['Weight0_unb_pes_uni'] +
                              x1[0] * x1[2] * x1[2] * df['Weight0_unb_pes_pes'] +
                              x1[0] * x1[2] * x1[3] * df['Weight0_unb_pes_sig'] +
                              x1[0] * x1[3] * df['Weight0_unb_sig'] +
                              x1[0] * x1[3] * x1[0] * df['Weight0_unb_sig_unb'] +
                              x1[0] * x1[3] * x1[1] * df['Weight0_unb_sig_uni'] +
                              x1[0] * x1[3] * x1[2] * df['Weight0_unb_sig_pes'] +
                              x1[0] * x1[3] * x1[3] * df['Weight0_unb_sig_sig'] +
                              x1[1] * df['Weight0_uni'] +
                              x1[1] * x1[0] * df['Weight0_uni_unb'] +
                              x1[1] * x1[0] * x1[0] * df['Weight0_uni_unb_unb'] +
                              x1[1] * x1[0] * x1[1] * df['Weight0_uni_unb_uni'] +
                              x1[1] * x1[0] * x1[2] * df['Weight0_uni_unb_pes'] +
                              x1[1] * x1[0] * x1[3] * df['Weight0_uni_unb_sig'] +
                              x1[1] * x1[1] * df['Weight0_uni_uni'] +
                              x1[1] * x1[1] * x1[0] * df['Weight0_uni_uni_unb'] +
                              x1[1] * x1[1] * x1[1] * df['Weight0_uni_uni_uni'] +
                              x1[1] * x1[1] * x1[2] * df['Weight0_uni_uni_pes'] +
                              x1[1] * x1[1] * x1[3] * df['Weight0_uni_uni_sig'] +
                              x1[1] * x1[2] * df['Weight0_uni_pes'] +
                              x1[1] * x1[2] * x1[0] * df['Weight0_uni_pes_unb'] +
                              x1[1] * x1[2] * x1[1] * df['Weight0_uni_pes_uni'] +
                              x1[1] * x1[2] * x1[2] * df['Weight0_uni_pes_pes'] +
                              x1[1] * x1[2] * x1[3] * df['Weight0_uni_pes_sig'] +
                              x1[1] * x1[3] * df['Weight0_uni_sig'] +
                              x1[1] * x1[3] * x1[0] * df['Weight0_uni_sig_unb'] +
                              x1[1] * x1[3] * x1[1] * df['Weight0_uni_sig_uni'] +
                              x1[1] * x1[3] * x1[2] * df['Weight0_uni_sig_pes'] +
                              x1[1] * x1[3] * x1[3] * df['Weight0_uni_sig_sig'] +
                              x1[2] * df['Weight0_pes'] +
                              x1[2] * x1[0] * df['Weight0_pes_unb'] +
                              x1[2] * x1[0] * x1[0] * df['Weight0_pes_unb_unb'] +
                              x1[2] * x1[0] * x1[1] * df['Weight0_pes_unb_uni'] +
                              x1[2] * x1[0] * x1[2] * df['Weight0_pes_unb_pes'] +
                              x1[2] * x1[0] * x1[3] * df['Weight0_pes_unb_sig'] +
                              x1[2] * x1[1] * df['Weight0_pes_uni'] +
                              x1[2] * x1[1] * x1[0] * df['Weight0_pes_uni_unb'] +
                              x1[2] * x1[1] * x1[1] * df['Weight0_pes_uni_uni'] +
                              x1[2] * x1[1] * x1[2] * df['Weight0_pes_uni_pes'] +
                              x1[2] * x1[1] * x1[3] * df['Weight0_pes_uni_sig'] +
                              x1[2] * x1[2] * df['Weight0_pes_pes'] +
                              x1[2] * x1[2] * x1[0] * df['Weight0_pes_pes_unb'] +
                              x1[2] * x1[2] * x1[1] * df['Weight0_pes_pes_uni'] +
                              x1[2] * x1[2] * x1[2] * df['Weight0_pes_pes_pes'] +
                              x1[2] * x1[2] * x1[3] * df['Weight0_pes_pes_sig'] +
                              x1[2] * x1[3] * df['Weight0_pes_sig'] +
                              x1[2] * x1[3] * x1[0] * df['Weight0_pes_sig_unb'] +
                              x1[2] * x1[3] * x1[1] * df['Weight0_pes_sig_uni'] +
                              x1[2] * x1[3] * x1[2] * df['Weight0_pes_sig_pes'] +
                              x1[2] * x1[3] * x1[3] * df['Weight0_pes_sig_sig'] +
                              x1[3] * df['Weight0_sig'] +
                              x1[3] * x1[0] * df['Weight0_sig_unb'] +
                              x1[3] * x1[0] * x1[0] * df['Weight0_sig_unb_unb'] +
                              x1[3] * x1[0] * x1[1] * df['Weight0_sig_unb_uni'] +
                              x1[3] * x1[0] * x1[2] * df['Weight0_sig_unb_pes'] +
                              x1[3] * x1[0] * x1[3] * df['Weight0_sig_unb_sig'] +
                              x1[3] * x1[1] * df['Weight0_sig_uni'] +
                              x1[3] * x1[1] * x1[0] * df['Weight0_sig_uni_unb'] +
                              x1[3] * x1[1] * x1[1] * df['Weight0_sig_uni_uni'] +
                              x1[3] * x1[1] * x1[2] * df['Weight0_sig_uni_pes'] +
                              x1[3] * x1[1] * x1[3] * df['Weight0_sig_uni_sig'] +
                              x1[3] * x1[2] * df['Weight0_sig_pes'] +
                              x1[3] * x1[2] * x1[0] * df['Weight0_sig_pes_unb'] +
                              x1[3] * x1[2] * x1[1] * df['Weight0_sig_pes_uni'] +
                              x1[3] * x1[2] * x1[2] * df['Weight0_sig_pes_pes'] +
                              x1[3] * x1[2] * x1[3] * df['Weight0_sig_pes_sig'] +
                              x1[3] * x1[3] * df['Weight0_sig_sig'] +
                              x1[3] * x1[3] * x1[0] * df['Weight0_sig_sig_unb'] +
                              x1[3] * x1[3] * x1[1] * df['Weight0_sig_sig_uni'] +
                              x1[3] * x1[3] * x1[2] * df['Weight0_sig_sig_pes'] +
                              x1[3] * x1[3] * x1[3] * df['Weight0_sig_sig_sig'] +
                              x2[0] * df['Weight_unb'] +
                              x2[0] * x2[0] * df['Weight_unb_unb'] +
                              x2[0] * x2[0] * x2[0] * df['Weight_unb_unb_unb'] +
                              x2[0] * x2[0] * x2[1] * df['Weight_unb_unb_uni'] +
                              x2[0] * x2[0] * x2[2] * df['Weight_unb_unb_pes'] +
                              x2[0] * x2[0] * x2[3] * df['Weight_unb_unb_sig'] +
                              x2[0] * x2[1] * df['Weight_unb_uni'] +
                              x2[0] * x2[1] * x2[0] * df['Weight_unb_uni_unb'] +
                              x2[0] * x2[1] * x2[1] * df['Weight_unb_uni_uni'] +
                              x2[0] * x2[1] * x2[2] * df['Weight_unb_uni_pes'] +
                              x2[0] * x2[1] * x2[3] * df['Weight_unb_uni_sig'] +
                              x2[0] * x2[2] * df['Weight_unb_pes'] +
                              x2[0] * x2[2] * x2[0] * df['Weight_unb_pes_unb'] +
                              x2[0] * x2[2] * x2[1] * df['Weight_unb_pes_uni'] +
                              x2[0] * x2[2] * x2[2] * df['Weight_unb_pes_pes'] +
                              x2[0] * x2[2] * x2[3] * df['Weight_unb_pes_sig'] +
                              x2[0] * x2[3] * df['Weight_unb_sig'] +
                              x2[0] * x2[3] * x2[0] * df['Weight_unb_sig_unb'] +
                              x2[0] * x2[3] * x2[1] * df['Weight_unb_sig_uni'] +
                              x2[0] * x2[3] * x2[2] * df['Weight_unb_sig_pes'] +
                              x2[0] * x2[3] * x2[3] * df['Weight_unb_sig_sig'] +
                              x2[1] * df['Weight_uni'] +
                              x2[1] * x2[0] * df['Weight_uni_unb'] +
                              x2[1] * x2[0] * x2[0] * df['Weight_uni_unb_unb'] +
                              x2[1] * x2[0] * x2[1] * df['Weight_uni_unb_uni'] +
                              x2[1] * x2[0] * x2[2] * df['Weight_uni_unb_pes'] +
                              x2[1] * x2[0] * x2[3] * df['Weight_uni_unb_sig'] +
                              x2[1] * x2[1] * df['Weight_uni_uni'] +
                              x2[1] * x2[1] * x2[0] * df['Weight_uni_uni_unb'] +
                              x2[1] * x2[1] * x2[1] * df['Weight_uni_uni_uni'] +
                              x2[1] * x2[1] * x2[2] * df['Weight_uni_uni_pes'] +
                              x2[1] * x2[1] * x2[3] * df['Weight_uni_uni_sig'] +
                              x2[1] * x2[2] * df['Weight_uni_pes'] +
                              x2[1] * x2[2] * x2[0] * df['Weight_uni_pes_unb'] +
                              x2[1] * x2[2] * x2[1] * df['Weight_uni_pes_uni'] +
                              x2[1] * x2[2] * x2[2] * df['Weight_uni_pes_pes'] +
                              x2[1] * x2[2] * x2[3] * df['Weight_uni_pes_sig'] +
                              x2[1] * x2[3] * df['Weight_uni_sig'] +
                              x2[1] * x2[3] * x2[0] * df['Weight_uni_sig_unb'] +
                              x2[1] * x2[3] * x2[1] * df['Weight_uni_sig_uni'] +
                              x2[1] * x2[3] * x2[2] * df['Weight_uni_sig_pes'] +
                              x2[1] * x2[3] * x2[3] * df['Weight_uni_sig_sig'] +
                              x2[2] * df['Weight_pes'] +
                              x2[2] * x2[0] * df['Weight_pes_unb'] +
                              x2[2] * x2[0] * x2[0] * df['Weight_pes_unb_unb'] +
                              x2[2] * x2[0] * x2[1] * df['Weight_pes_unb_uni'] +
                              x2[2] * x2[0] * x2[2] * df['Weight_pes_unb_pes'] +
                              x2[2] * x2[0] * x2[3] * df['Weight_pes_unb_sig'] +
                              x2[2] * x2[1] * df['Weight_pes_uni'] +
                              x2[2] * x2[1] * x2[0] * df['Weight_pes_uni_unb'] +
                              x2[2] * x2[1] * x2[1] * df['Weight_pes_uni_uni'] +
                              x2[2] * x2[1] * x2[2] * df['Weight_pes_uni_pes'] +
                              x2[2] * x2[1] * x2[3] * df['Weight_pes_uni_sig'] +
                              x2[2] * x2[2] * df['Weight_pes_pes'] +
                              x2[2] * x2[2] * x2[0] * df['Weight_pes_pes_unb'] +
                              x2[2] * x2[2] * x2[1] * df['Weight_pes_pes_uni'] +
                              x2[2] * x2[2] * x2[2] * df['Weight_pes_pes_pes'] +
                              x2[2] * x2[2] * x2[3] * df['Weight_pes_pes_sig'] +
                              x2[2] * x2[3] * df['Weight_pes_sig'] +
                              x2[2] * x2[3] * x2[0] * df['Weight_pes_sig_unb'] +
                              x2[2] * x2[3] * x2[1] * df['Weight_pes_sig_uni'] +
                              x2[2] * x2[3] * x2[2] * df['Weight_pes_sig_pes'] +
                              x2[2] * x2[3] * x2[3] * df['Weight_pes_sig_sig'] +
                              x2[3] * df['Weight_sig'] +
                              x2[3] * x2[0] * df['Weight_sig_unb'] +
                              x2[3] * x2[0] * x2[0] * df['Weight_sig_unb_unb'] +
                              x2[3] * x2[0] * x2[1] * df['Weight_sig_unb_uni'] +
                              x2[3] * x2[0] * x2[2] * df['Weight_sig_unb_pes'] +
                              x2[3] * x2[0] * x2[3] * df['Weight_sig_unb_sig'] +
                              x2[3] * x2[1] * df['Weight_sig_uni'] +
                              x2[3] * x2[1] * x2[0] * df['Weight_sig_uni_unb'] +
                              x2[3] * x2[1] * x2[1] * df['Weight_sig_uni_uni'] +
                              x2[3] * x2[1] * x2[2] * df['Weight_sig_uni_pes'] +
                              x2[3] * x2[1] * x2[3] * df['Weight_sig_uni_sig'] +
                              x2[3] * x2[2] * df['Weight_sig_pes'] +
                              x2[3] * x2[2] * x2[0] * df['Weight_sig_pes_unb'] +
                              x2[3] * x2[2] * x2[1] * df['Weight_sig_pes_uni'] +
                              x2[3] * x2[2] * x2[2] * df['Weight_sig_pes_pes'] +
                              x2[3] * x2[2] * x2[3] * df['Weight_sig_pes_sig'] +
                              x2[3] * x2[3] * df['Weight_sig_sig'] +
                              x2[3] * x2[3] * x2[0] * df['Weight_sig_sig_unb'] +
                              x2[3] * x2[3] * x2[1] * df['Weight_sig_sig_uni'] +
                              x2[3] * x2[3] * x2[2] * df['Weight_sig_sig_pes'] +
                              x2[3] * x2[3] * x2[3] * df['Weight_sig_sig_sig'] +
                              x3[0] * df['Weight_unb'] +
                              x3[0] * x3[0] * df['Weight_unb_unb'] +
                              x3[0] * x3[0] * x3[0] * df['Weight_unb_unb_unb'] +
                              x3[0] * x3[0] * x3[1] * df['Weight_unb_unb_uni'] +
                              x3[0] * x3[0] * x3[2] * df['Weight_unb_unb_pes'] +
                              x3[0] * x3[0] * x3[3] * df['Weight_unb_unb_sig'] +
                              x3[0] * x3[1] * df['Weight_unb_uni'] +
                              x3[0] * x3[1] * x3[0] * df['Weight_unb_uni_unb'] +
                              x3[0] * x3[1] * x3[1] * df['Weight_unb_uni_uni'] +
                              x3[0] * x3[1] * x3[2] * df['Weight_unb_uni_pes'] +
                              x3[0] * x3[1] * x3[3] * df['Weight_unb_uni_sig'] +
                              x3[0] * x3[2] * df['Weight_unb_pes'] +
                              x3[0] * x3[2] * x3[0] * df['Weight_unb_pes_unb'] +
                              x3[0] * x3[2] * x3[1] * df['Weight_unb_pes_uni'] +
                              x3[0] * x3[2] * x3[2] * df['Weight_unb_pes_pes'] +
                              x3[0] * x3[2] * x3[3] * df['Weight_unb_pes_sig'] +
                              x3[0] * x3[3] * df['Weight_unb_sig'] +
                              x3[0] * x3[3] * x3[0] * df['Weight_unb_sig_unb'] +
                              x3[0] * x3[3] * x3[1] * df['Weight_unb_sig_uni'] +
                              x3[0] * x3[3] * x3[2] * df['Weight_unb_sig_pes'] +
                              x3[0] * x3[3] * x3[3] * df['Weight_unb_sig_sig'] +
                              x3[1] * df['Weight_uni'] +
                              x3[1] * x3[0] * df['Weight_uni_unb'] +
                              x3[1] * x3[0] * x3[0] * df['Weight_uni_unb_unb'] +
                              x3[1] * x3[0] * x3[1] * df['Weight_uni_unb_uni'] +
                              x3[1] * x3[0] * x3[2] * df['Weight_uni_unb_pes'] +
                              x3[1] * x3[0] * x3[3] * df['Weight_uni_unb_sig'] +
                              x3[1] * x3[1] * df['Weight_uni_uni'] +
                              x3[1] * x3[1] * x3[0] * df['Weight_uni_uni_unb'] +
                              x3[1] * x3[1] * x3[1] * df['Weight_uni_uni_uni'] +
                              x3[1] * x3[1] * x3[2] * df['Weight_uni_uni_pes'] +
                              x3[1] * x3[1] * x3[3] * df['Weight_uni_uni_sig'] +
                              x3[1] * x3[2] * df['Weight_uni_pes'] +
                              x3[1] * x3[2] * x3[0] * df['Weight_uni_pes_unb'] +
                              x3[1] * x3[2] * x3[1] * df['Weight_uni_pes_uni'] +
                              x3[1] * x3[2] * x3[2] * df['Weight_uni_pes_pes'] +
                              x3[1] * x3[2] * x3[3] * df['Weight_uni_pes_sig'] +
                              x3[1] * x3[3] * df['Weight_uni_sig'] +
                              x3[1] * x3[3] * x3[0] * df['Weight_uni_sig_unb'] +
                              x3[1] * x3[3] * x3[1] * df['Weight_uni_sig_uni'] +
                              x3[1] * x3[3] * x3[2] * df['Weight_uni_sig_pes'] +
                              x3[1] * x3[3] * x3[3] * df['Weight_uni_sig_sig'] +
                              x3[2] * df['Weight_pes'] +
                              x3[2] * x3[0] * df['Weight_pes_unb'] +
                              x3[2] * x3[0] * x3[0] * df['Weight_pes_unb_unb'] +
                              x3[2] * x3[0] * x3[1] * df['Weight_pes_unb_uni'] +
                              x3[2] * x3[0] * x3[2] * df['Weight_pes_unb_pes'] +
                              x3[2] * x3[0] * x3[3] * df['Weight_pes_unb_sig'] +
                              x3[2] * x3[1] * df['Weight_pes_uni'] +
                              x3[2] * x3[1] * x3[0] * df['Weight_pes_uni_unb'] +
                              x3[2] * x3[1] * x3[1] * df['Weight_pes_uni_uni'] +
                              x3[2] * x3[1] * x3[2] * df['Weight_pes_uni_pes'] +
                              x3[2] * x3[1] * x3[3] * df['Weight_pes_uni_sig'] +
                              x3[2] * x3[2] * df['Weight_pes_pes'] +
                              x3[2] * x3[2] * x3[0] * df['Weight_pes_pes_unb'] +
                              x3[2] * x3[2] * x3[1] * df['Weight_pes_pes_uni'] +
                              x3[2] * x3[2] * x3[2] * df['Weight_pes_pes_pes'] +
                              x3[2] * x3[2] * x3[3] * df['Weight_pes_pes_sig'] +
                              x3[2] * x3[3] * df['Weight_pes_sig'] +
                              x3[2] * x3[3] * x3[0] * df['Weight_pes_sig_unb'] +
                              x3[2] * x3[3] * x3[1] * df['Weight_pes_sig_uni'] +
                              x3[2] * x3[3] * x3[2] * df['Weight_pes_sig_pes'] +
                              x3[2] * x3[3] * x3[3] * df['Weight_pes_sig_sig'] +
                              x3[3] * df['Weight_sig'] +
                              x3[3] * x3[0] * df['Weight_sig_unb'] +
                              x3[3] * x3[0] * x3[0] * df['Weight_sig_unb_unb'] +
                              x3[3] * x3[0] * x3[1] * df['Weight_sig_unb_uni'] +
                              x3[3] * x3[0] * x3[2] * df['Weight_sig_unb_pes'] +
                              x3[3] * x3[0] * x3[3] * df['Weight_sig_unb_sig'] +
                              x3[3] * x3[1] * df['Weight_sig_uni'] +
                              x3[3] * x3[1] * x3[0] * df['Weight_sig_uni_unb'] +
                              x3[3] * x3[1] * x3[1] * df['Weight_sig_uni_uni'] +
                              x3[3] * x3[1] * x3[2] * df['Weight_sig_uni_pes'] +
                              x3[3] * x3[1] * x3[3] * df['Weight_sig_uni_sig'] +
                              x3[3] * x3[2] * df['Weight_sig_pes'] +
                              x3[3] * x3[2] * x3[0] * df['Weight_sig_pes_unb'] +
                              x3[3] * x3[2] * x3[1] * df['Weight_sig_pes_uni'] +
                              x3[3] * x3[2] * x3[2] * df['Weight_sig_pes_pes'] +
                              x3[3] * x3[2] * x3[3] * df['Weight_sig_pes_sig'] +
                              x3[3] * x3[3] * df['Weight_sig_sig'] +
                              x3[3] * x3[3] * x3[0] * df['Weight_sig_sig_unb'] +
                              x3[3] * x3[3] * x3[1] * df['Weight_sig_sig_uni'] +
                              x3[3] * x3[3] * x3[2] * df['Weight_sig_sig_pes'] +
                              x3[3] * x3[3] * x3[3] * df['Weight_sig_sig_sig'] +
                              x4[0] * df['Weight_unb'] +
                              x4[0] * x4[0] * df['Weight_unb_unb'] +
                              x4[0] * x4[0] * x4[0] * df['Weight_unb_unb_unb'] +
                              x4[0] * x4[0] * x4[1] * df['Weight_unb_unb_uni'] +
                              x4[0] * x4[0] * x4[2] * df['Weight_unb_unb_pes'] +
                              x4[0] * x4[0] * x4[3] * df['Weight_unb_unb_sig'] +
                              x4[0] * x4[1] * df['Weight_unb_uni'] +
                              x4[0] * x4[1] * x4[0] * df['Weight_unb_uni_unb'] +
                              x4[0] * x4[1] * x4[1] * df['Weight_unb_uni_uni'] +
                              x4[0] * x4[1] * x4[2] * df['Weight_unb_uni_pes'] +
                              x4[0] * x4[1] * x4[3] * df['Weight_unb_uni_sig'] +
                              x4[0] * x4[2] * df['Weight_unb_pes'] +
                              x4[0] * x4[2] * x4[0] * df['Weight_unb_pes_unb'] +
                              x4[0] * x4[2] * x4[1] * df['Weight_unb_pes_uni'] +
                              x4[0] * x4[2] * x4[2] * df['Weight_unb_pes_pes'] +
                              x4[0] * x4[2] * x4[3] * df['Weight_unb_pes_sig'] +
                              x4[0] * x4[3] * df['Weight_unb_sig'] +
                              x4[0] * x4[3] * x4[0] * df['Weight_unb_sig_unb'] +
                              x4[0] * x4[3] * x4[1] * df['Weight_unb_sig_uni'] +
                              x4[0] * x4[3] * x4[2] * df['Weight_unb_sig_pes'] +
                              x4[0] * x4[3] * x4[3] * df['Weight_unb_sig_sig'] +
                              x4[1] * df['Weight_uni'] +
                              x4[1] * x4[0] * df['Weight_uni_unb'] +
                              x4[1] * x4[0] * x4[0] * df['Weight_uni_unb_unb'] +
                              x4[1] * x4[0] * x4[1] * df['Weight_uni_unb_uni'] +
                              x4[1] * x4[0] * x4[2] * df['Weight_uni_unb_pes'] +
                              x4[1] * x4[0] * x4[3] * df['Weight_uni_unb_sig'] +
                              x4[1] * x4[1] * df['Weight_uni_uni'] +
                              x4[1] * x4[1] * x4[0] * df['Weight_uni_uni_unb'] +
                              x4[1] * x4[1] * x4[1] * df['Weight_uni_uni_uni'] +
                              x4[1] * x4[1] * x4[2] * df['Weight_uni_uni_pes'] +
                              x4[1] * x4[1] * x4[3] * df['Weight_uni_uni_sig'] +
                              x4[1] * x4[2] * df['Weight_uni_pes'] +
                              x4[1] * x4[2] * x4[0] * df['Weight_uni_pes_unb'] +
                              x4[1] * x4[2] * x4[1] * df['Weight_uni_pes_uni'] +
                              x4[1] * x4[2] * x4[2] * df['Weight_uni_pes_pes'] +
                              x4[1] * x4[2] * x4[3] * df['Weight_uni_pes_sig'] +
                              x4[1] * x4[3] * df['Weight_uni_sig'] +
                              x4[1] * x4[3] * x4[0] * df['Weight_uni_sig_unb'] +
                              x4[1] * x4[3] * x4[1] * df['Weight_uni_sig_uni'] +
                              x4[1] * x4[3] * x4[2] * df['Weight_uni_sig_pes'] +
                              x4[1] * x4[3] * x4[3] * df['Weight_uni_sig_sig'] +
                              x4[2] * df['Weight_pes'] +
                              x4[2] * x4[0] * df['Weight_pes_unb'] +
                              x4[2] * x4[0] * x4[0] * df['Weight_pes_unb_unb'] +
                              x4[2] * x4[0] * x4[1] * df['Weight_pes_unb_uni'] +
                              x4[2] * x4[0] * x4[2] * df['Weight_pes_unb_pes'] +
                              x4[2] * x4[0] * x4[3] * df['Weight_pes_unb_sig'] +
                              x4[2] * x4[1] * df['Weight_pes_uni'] +
                              x4[2] * x4[1] * x4[0] * df['Weight_pes_uni_unb'] +
                              x4[2] * x4[1] * x4[1] * df['Weight_pes_uni_uni'] +
                              x4[2] * x4[1] * x4[2] * df['Weight_pes_uni_pes'] +
                              x4[2] * x4[1] * x4[3] * df['Weight_pes_uni_sig'] +
                              x4[2] * x4[2] * df['Weight_pes_pes'] +
                              x4[2] * x4[2] * x4[0] * df['Weight_pes_pes_unb'] +
                              x4[2] * x4[2] * x4[1] * df['Weight_pes_pes_uni'] +
                              x4[2] * x4[2] * x4[2] * df['Weight_pes_pes_pes'] +
                              x4[2] * x4[2] * x4[3] * df['Weight_pes_pes_sig'] +
                              x4[2] * x4[3] * df['Weight_pes_sig'] +
                              x4[2] * x4[3] * x4[0] * df['Weight_pes_sig_unb'] +
                              x4[2] * x4[3] * x4[1] * df['Weight_pes_sig_uni'] +
                              x4[2] * x4[3] * x4[2] * df['Weight_pes_sig_pes'] +
                              x4[2] * x4[3] * x4[3] * df['Weight_pes_sig_sig'] +
                              x4[3] * df['Weight_sig'] +
                              x4[3] * x4[0] * df['Weight_sig_unb'] +
                              x4[3] * x4[0] * x4[0] * df['Weight_sig_unb_unb'] +
                              x4[3] * x4[0] * x4[1] * df['Weight_sig_unb_uni'] +
                              x4[3] * x4[0] * x4[2] * df['Weight_sig_unb_pes'] +
                              x4[3] * x4[0] * x4[3] * df['Weight_sig_unb_sig'] +
                              x4[3] * x4[1] * df['Weight_sig_uni'] +
                              x4[3] * x4[1] * x4[0] * df['Weight_sig_uni_unb'] +
                              x4[3] * x4[1] * x4[1] * df['Weight_sig_uni_uni'] +
                              x4[3] * x4[1] * x4[2] * df['Weight_sig_uni_pes'] +
                              x4[3] * x4[1] * x4[3] * df['Weight_sig_uni_sig'] +
                              x4[3] * x4[2] * df['Weight_sig_pes'] +
                              x4[3] * x4[2] * x4[0] * df['Weight_sig_pes_unb'] +
                              x4[3] * x4[2] * x4[1] * df['Weight_sig_pes_uni'] +
                              x4[3] * x4[2] * x4[2] * df['Weight_sig_pes_pes'] +
                              x4[3] * x4[2] * x4[3] * df['Weight_sig_pes_sig'] +
                              x4[3] * x4[3] * df['Weight_sig_sig'] +
                              x4[3] * x4[3] * x4[0] * df['Weight_sig_sig_unb'] +
                              x4[3] * x4[3] * x4[1] * df['Weight_sig_sig_uni'] +
                              x4[3] * x4[3] * x4[2] * df['Weight_sig_sig_pes'] +
                              x4[3] * x4[3] * x4[3] * df['Weight_sig_sig_sig'] +
                              x5[0] * df['Weight_unb'] +
                              x5[0] * x5[0] * df['Weight_unb_unb'] +
                              x5[0] * x5[0] * x5[0] * df['Weight_unb_unb_unb'] +
                              x5[0] * x5[0] * x5[1] * df['Weight_unb_unb_uni'] +
                              x5[0] * x5[0] * x5[2] * df['Weight_unb_unb_pes'] +
                              x5[0] * x5[0] * x5[3] * df['Weight_unb_unb_sig'] +
                              x5[0] * x5[1] * df['Weight_unb_uni'] +
                              x5[0] * x5[1] * x5[0] * df['Weight_unb_uni_unb'] +
                              x5[0] * x5[1] * x5[1] * df['Weight_unb_uni_uni'] +
                              x5[0] * x5[1] * x5[2] * df['Weight_unb_uni_pes'] +
                              x5[0] * x5[1] * x5[3] * df['Weight_unb_uni_sig'] +
                              x5[0] * x5[2] * df['Weight_unb_pes'] +
                              x5[0] * x5[2] * x5[0] * df['Weight_unb_pes_unb'] +
                              x5[0] * x5[2] * x5[1] * df['Weight_unb_pes_uni'] +
                              x5[0] * x5[2] * x5[2] * df['Weight_unb_pes_pes'] +
                              x5[0] * x5[2] * x5[3] * df['Weight_unb_pes_sig'] +
                              x5[0] * x5[3] * df['Weight_unb_sig'] +
                              x5[0] * x5[3] * x5[0] * df['Weight_unb_sig_unb'] +
                              x5[0] * x5[3] * x5[1] * df['Weight_unb_sig_uni'] +
                              x5[0] * x5[3] * x5[2] * df['Weight_unb_sig_pes'] +
                              x5[0] * x5[3] * x5[3] * df['Weight_unb_sig_sig'] +
                              x5[1] * df['Weight_uni'] +
                              x5[1] * x5[0] * df['Weight_uni_unb'] +
                              x5[1] * x5[0] * x5[0] * df['Weight_uni_unb_unb'] +
                              x5[1] * x5[0] * x5[1] * df['Weight_uni_unb_uni'] +
                              x5[1] * x5[0] * x5[2] * df['Weight_uni_unb_pes'] +
                              x5[1] * x5[0] * x5[3] * df['Weight_uni_unb_sig'] +
                              x5[1] * x5[1] * df['Weight_uni_uni'] +
                              x5[1] * x5[1] * x5[0] * df['Weight_uni_uni_unb'] +
                              x5[1] * x5[1] * x5[1] * df['Weight_uni_uni_uni'] +
                              x5[1] * x5[1] * x5[2] * df['Weight_uni_uni_pes'] +
                              x5[1] * x5[1] * x5[3] * df['Weight_uni_uni_sig'] +
                              x5[1] * x5[2] * df['Weight_uni_pes'] +
                              x5[1] * x5[2] * x5[0] * df['Weight_uni_pes_unb'] +
                              x5[1] * x5[2] * x5[1] * df['Weight_uni_pes_uni'] +
                              x5[1] * x5[2] * x5[2] * df['Weight_uni_pes_pes'] +
                              x5[1] * x5[2] * x5[3] * df['Weight_uni_pes_sig'] +
                              x5[1] * x5[3] * df['Weight_uni_sig'] +
                              x5[1] * x5[3] * x5[0] * df['Weight_uni_sig_unb'] +
                              x5[1] * x5[3] * x5[1] * df['Weight_uni_sig_uni'] +
                              x5[1] * x5[3] * x5[2] * df['Weight_uni_sig_pes'] +
                              x5[1] * x5[3] * x5[3] * df['Weight_uni_sig_sig'] +
                              x5[2] * df['Weight_pes'] +
                              x5[2] * x5[0] * df['Weight_pes_unb'] +
                              x5[2] * x5[0] * x5[0] * df['Weight_pes_unb_unb'] +
                              x5[2] * x5[0] * x5[1] * df['Weight_pes_unb_uni'] +
                              x5[2] * x5[0] * x5[2] * df['Weight_pes_unb_pes'] +
                              x5[2] * x5[0] * x5[3] * df['Weight_pes_unb_sig'] +
                              x5[2] * x5[1] * df['Weight_pes_uni'] +
                              x5[2] * x5[1] * x5[0] * df['Weight_pes_uni_unb'] +
                              x5[2] * x5[1] * x5[1] * df['Weight_pes_uni_uni'] +
                              x5[2] * x5[1] * x5[2] * df['Weight_pes_uni_pes'] +
                              x5[2] * x5[1] * x5[3] * df['Weight_pes_uni_sig'] +
                              x5[2] * x5[2] * df['Weight_pes_pes'] +
                              x5[2] * x5[2] * x5[0] * df['Weight_pes_pes_unb'] +
                              x5[2] * x5[2] * x5[1] * df['Weight_pes_pes_uni'] +
                              x5[2] * x5[2] * x5[2] * df['Weight_pes_pes_pes'] +
                              x5[2] * x5[2] * x5[3] * df['Weight_pes_pes_sig'] +
                              x5[2] * x5[3] * df['Weight_pes_sig'] +
                              x5[2] * x5[3] * x5[0] * df['Weight_pes_sig_unb'] +
                              x5[2] * x5[3] * x5[1] * df['Weight_pes_sig_uni'] +
                              x5[2] * x5[3] * x5[2] * df['Weight_pes_sig_pes'] +
                              x5[2] * x5[3] * x5[3] * df['Weight_pes_sig_sig'] +
                              x5[3] * df['Weight_sig'] +
                              x5[3] * x5[0] * df['Weight_sig_unb'] +
                              x5[3] * x5[0] * x5[0] * df['Weight_sig_unb_unb'] +
                              x5[3] * x5[0] * x5[1] * df['Weight_sig_unb_uni'] +
                              x5[3] * x5[0] * x5[2] * df['Weight_sig_unb_pes'] +
                              x5[3] * x5[0] * x5[3] * df['Weight_sig_unb_sig'] +
                              x5[3] * x5[1] * df['Weight_sig_uni'] +
                              x5[3] * x5[1] * x5[0] * df['Weight_sig_uni_unb'] +
                              x5[3] * x5[1] * x5[1] * df['Weight_sig_uni_uni'] +
                              x5[3] * x5[1] * x5[2] * df['Weight_sig_uni_pes'] +
                              x5[3] * x5[1] * x5[3] * df['Weight_sig_uni_sig'] +
                              x5[3] * x5[2] * df['Weight_sig_pes'] +
                              x5[3] * x5[2] * x5[0] * df['Weight_sig_pes_unb'] +
                              x5[3] * x5[2] * x5[1] * df['Weight_sig_pes_uni'] +
                              x5[3] * x5[2] * x5[2] * df['Weight_sig_pes_pes'] +
                              x5[3] * x5[2] * x5[3] * df['Weight_sig_pes_sig'] +
                              x5[3] * x5[3] * df['Weight_sig_sig'] +
                              x5[3] * x5[3] * x5[0] * df['Weight_sig_sig_unb'] +
                              x5[3] * x5[3] * x5[1] * df['Weight_sig_sig_uni'] +
                              x5[3] * x5[3] * x5[2] * df['Weight_sig_sig_pes'] +
                              x5[3] * x5[3] * x5[3] * df['Weight_sig_sig_sig']
                                                      )/5)


def find_best_probs(train, label, c13k=True):
    if c13k:
        return minimize(fun=MSE_biases, x0=np.array([0.4927, 0.1691, 0.1691, 0.1691,0.6407, 0.1198, 0.1198, 0.1198,0.6793, 0.1069, 0.1069, 0.1069,0.7036, 0.0988, 0.0988, 0.0988,0.7211, 0.0930, 0.0930, 0.0930]),
                        args=(train, label),
                        bounds=[(0., 1), (0., 1), (0., 1), (0., 1),(0., 1), (0., 1), (0., 1), (0., 1),(0., 1), (0., 1), (0., 1), (0., 1),(0., 1), (0., 1), (0., 1), (0., 1),(0., 1), (0., 1), (0., 1), (0., 1)],
                        constraints=({'type': 'eq', 'fun': lambda b: 1 - sum(b[0:4])},{'type': 'eq', 'fun': lambda b: 1 - sum(b[4:8])},{'type': 'eq', 'fun': lambda b: 1 - sum(b[8:12])},{'type': 'eq', 'fun': lambda b: 1 - sum(b[12:16])},{'type': 'eq', 'fun': lambda b: 1 - sum(b[16:20])}), method='slsqp').x
    else:
        x1 = minimize(fun=MSE_biases, x0=np.array([0.25, 0.25, 0.25, 0.25]), args=(train[train['block'] == 1], label),
                      bounds=[(0., 1), (0., 1), (0., 1), (0., 1)],
                      constraints=({'type': 'eq', 'fun': lambda b: 1 - sum(b)},), method='slsqp').x
        x2 = minimize(fun=MSE_biases, x0=np.array([0.25, 0.25, 0.25, 0.25]), args=(train[train['block'] == 2], label),
                      bounds=[(0., 1), (0., 1), (0., 1), (0., 1)],
                      constraints=({'type': 'eq', 'fun': lambda b: 1 - sum(b)},), method='slsqp').x
        x3 = minimize(fun=MSE_biases, x0=np.array([0.25, 0.25, 0.25, 0.25]), args=(train[train['block'] == 3], label),
                      bounds=[(0., 1), (0., 1), (0., 1), (0., 1)],
                      constraints=({'type': 'eq', 'fun': lambda b: 1 - sum(b)},), method='slsqp').x
        x4 = minimize(fun=MSE_biases, x0=np.array([0.25, 0.25, 0.25, 0.25]), args=(train[train['block'] == 4], label),
                      bounds=[(0., 1), (0., 1), (0., 1), (0., 1)],
                      constraints=({'type': 'eq', 'fun': lambda b: 1 - sum(b)},), method='slsqp').x
        x5 = minimize(fun=MSE_biases, x0=np.array([0.25, 0.25, 0.25, 0.25]), args=(train[train['block'] == 5], label),
                      bounds=[(0., 1), (0., 1), (0., 1), (0., 1)],
                      constraints=({'type': 'eq', 'fun': lambda b: 1 - sum(b)},), method='slsqp').x
        return x1, x2, x3, x4, x5


if __name__ == '__main__':
    add_bias_weights(pd.read_csv('c13k_selections.csv'), 'c13k_selections_ST')
