import os
import logging
#os.chdir('---')
#####################################################################################
### Section A: Please change this section to import necessary files and packages ###
#####################################################################################
import pandas as pd
import numpy as np
import time
from CPC15_BEASTpred import CPC15_BEAST_individual_pred

import time
import multiprocessing 

suffix = '10K_multiprocess'
data_read = '10k'

def create_Beast_pred(Data, nProblems_i, i) : 
    
    ## Create the Ground truth for every participant and problem ie every line 

    for game in nProblems_i:
        # read problem's parameters
        Ha = Data.loc[Data.GameId == game, 'Ha'].values[0]
        pHa = Data.loc[Data.GameId == game, 'pHa'].values[0]
        La = Data.loc[Data.GameId == game, 'La'].values[0]
        LotShapeA = Data.loc[Data.GameId == game, 'LotShapeA'].values[0]
        LotNumA = Data.loc[Data.GameId == game, 'LotNumA'].values[0]
        Hb = Data.loc[Data.GameId == game, 'Hb'].values[0]
        pHb = Data.loc[Data.GameId == game]['pHb'].values[0]
        Lb = Data.loc[Data.GameId == game]['Lb'].values[0]
        LotShapeB = Data.loc[Data.GameId == game]['LotShapeB'].values[0]
        LotNumB = Data.loc[Data.GameId == game]['LotNumB'].values[0]
        Amb = Data.loc[Data.GameId == game]['Amb'].values[0]
        Corr = Data.loc[Data.GameId == game]['Corr'].values[0]
        
        
        # Use Beast to give the ground thruth : modify beast : only 50 simulations for every prediction not 50000
        (all_prediction, Prediction) = CPC15_BEAST_individual_pred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)

        #Add the prediction to the Data
        Data.loc[Data.GameId == game, 'Beast1'] = Prediction[0]

        # Replicate the Problem 200 times, for 200 participants in each problem 
        dfdf = Data.loc[(Data.GameId == game)] 
        rep_prob = pd.DataFrame(np.repeat(dfdf.values, int(200*3)-1, axis = 0), columns= Data.columns)
#         rep_prob = pd.DataFrame(np.repeat(dfdf.values, int(200*54)-1, axis = 0), columns= Data.columns)

#         Data = pd.concat([Data, rep_prob])
        Data = Data.append(rep_prob, ignore_index = True) 
        all_pred_df= pd.DataFrame.from_records(all_prediction)
        all_pred_df.columns = ['B1', 'sigma', 'kapa', 'beta', 'gama', 'psi', 'theta', 'wamb']
        for col in ['B1', 'sigma', 'kapa', 'beta', 'gama', 'psi', 'theta', 'wamb'] : 
            Data.loc[(Data['GameId'] == game), col] = pd.Series(list(all_pred_df[col])).values
        
        # for verbose progression
        if game % 5 == 0 : 
            logging.info('{}: Finish indivdual_problem number: {}{}'.format((time.asctime(time.localtime(time.time()))), game, Data.shape))
    
    logging.info("Saving data to file ")
    # Save Data with Beast Individual Predictions 
    Data.to_csv(str("/home/meghanmergui/Synth20/Create_Synthetic_Games/3_Generate_Answers /synth_all_kapa/synth_" + str(i) + ".csv"), index = False)
    logging.info('data saved')
        


if __name__ == '__main__':
    
    logging.basicConfig(filename=str('Beast18_' + str(suffix) + '.log'), level=logging.INFO)
    logging.info('Started')

    # load problems to predict 
    Data = pd.read_csv('/home/meghanmergui/Synth20/Create_Synthetic_Games/1_games_10K/Synth10K2.csv')
    Data['GameId'] = range(1, len(Data) + 1)

    # Add the "ground thruth = Beast_individual" for every participant in evry block : 
    Data['B1'] = 0
    
    Data['sigma'] = 0
    Data['kapa'] = 0
    Data['beta']= 0 
    Data['gama'] = 0 
    Data['psi'] = 0
    Data['theta'] = 0
    Data['wamb'] = 0
    
    Data['Beast1'] = 0

    # Total number of Problem_individual to predict
    nProblems = Data['GameId'].unique()
    ### End of Section A ###
    
    starttime = time.time()
    processes = []
#     manager = multiprocessing.Manager()
#     return_dict = manager.dict()
    
    for i in range(0,20):
        nProblems_i = nProblems[i*500 : 500*(i+1)]
        p = multiprocessing.Process(target=create_Beast_pred, args=(Data.loc[Data.GameId.isin(nProblems_i)], nProblems_i, i))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()                             
        
    logging.info('That took {} seconds'.format(time.time() - starttime))
   
    
    ### End of Section C ###

    ########################################################
    ### Section D: Save the Beast Individual Predictions ###
    ########################################################

    
    ### End of Section D ###

    logging.info('Finished')

    ### End of Section D ###