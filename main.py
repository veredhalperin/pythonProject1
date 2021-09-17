import os
# os.chdir('---')
#####################################################################################
### Section A: Please change this section to import necessary files and packages ###
#####################################################################################
import pandas as pd
from CPC18PsychForestPython.get_PF_Features import get_PF_Features
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
logging.basicConfig(filename='main.log', level=logging.DEBUG)

if __name__ == '__main__':
    ####################################################
    ### Section B: Please do not change this section ###
    ####################################################
    # load problems to predict (in this example, the estimation set problems)
    Data = pd.read_csv('c13k_selections.csv')
    # useful variables
    nProblems = Data.shape[0]
    Data.index = range(nProblems)
    Problems, BEVas, BEVbs, STas, STbs = [], [], [], [], []
    ### End of Section A ###

    #################################################################
    ### Section C: Please change only lines 40-45 in this section ###
    #################################################################
    start=31
    end=32
    for prob in range(start, end):
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
        Problem = Data['Problem'][prob]
        tmpFeats, BEVa, BEVb, STa, STb = get_PF_Features(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB,
                                                         LotNumB, Amb, Corr)
        for column in Data.columns:
            tmpFeats[column] = Data[column][prob]
        if prob == start:
            data = tmpFeats
        else:
            data = data.append(tmpFeats)
        print(prob)
        logging.debug(str(prob))
        Problems.append(Problem)
        BEVas.append(BEVa)
        BEVbs.append(BEVb)
        STas.append(STa)
        STbs.append(STb)
        if (prob+1) % 100 == 0 or prob==end-1:
            dict = {}
            dict['Problem'] = Problems
            dict['BEVa'] = BEVas
            dict['BEVb'] = BEVbs
            for sim in range(4000):
                for trial in range(5):
                    STa = []
                    STb = []
                    for i in range(len(STas)):
                        STa.append(STas[i][sim][trial][0])
                        STb.append(STbs[i][sim][trial][0])
                    dict['STa_' + str(sim) + '_' + str(trial)] = STa
                    dict['STb_' + str(sim) + '_' + str(trial)] = STb
            pd.DataFrame.from_dict(dict).to_csv(str(prob) + '.csv', index=False)
            data.to_csv('c13k_selections2.csv', index=False)
            Problems, BEVas, BEVbs, STas, STbs = [], [], [], [], []
