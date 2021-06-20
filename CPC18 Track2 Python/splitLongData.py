import pandas as pd
import random
import numpy as np


def splitLongData(dd, seed=1, nSubjTest=30, nGamesPerSubjTest=5):

    if "SubjID" not in dd.columns or "GameID" not in dd.columns:
        print("data must include SubjID and GameID columns")
        return None, None

    dd = dd.sort_values(by=['SubjID', 'GameID'])
    np.random.seed(seed)
    subjs = np.array(list(dd['SubjID'].unique()))
    subj2remove = np.random.choice(subjs, nSubjTest, replace=False)
    test = pd.DataFrame()
    train = dd
    for s in range(0, nSubjTest):
        subjD = dd.loc[dd['SubjID'] == subj2remove[s]]
        games = np.array(list(subjD['GameID'].unique()))
        games2remove = np.random.choice(games, nGamesPerSubjTest, replace=False)
        for g in range(0, nGamesPerSubjTest):
            test = pd.concat([test, dd.loc[((dd['SubjID'] == subj2remove[s]) & (dd['GameID'] == games2remove[g]))]])
            train.drop(train[(train['SubjID'] == subj2remove[s]) & (train['GameID'] == games2remove[g])].index,
                       inplace=True)
            # train = train.loc[(train['SubjID'] != subj2remove[s]) & (train['GameID'] != games2remove[g]), :]

    return train, test
