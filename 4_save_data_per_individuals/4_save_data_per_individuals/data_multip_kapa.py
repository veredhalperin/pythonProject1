import pandas as pd
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.display.max_columns = 10000
import numpy as np
import logging
import multiprocessing
import time
from random import sample


## Add lot shape columns 
def lot_shape_convert(lot_shape):
    if lot_shape == '-' : 
        return 1,0,0,0
    elif lot_shape == 'Symm' : 
        return 0,1,0,0
    elif lot_shape == 'L-skew' : 
        return 0,0,1,0
    elif lot_shape == 'R-skew' :
        return 0,0,0,1


def create_MLPdata(num_subj, df_indivK, sigma, beta, kapa, gama): 
    df_all20 = pd.DataFrame(columns=x_col+y_col+['SubjID'])

    logging.info('{}: shape : {}, process {}'.format((time.asctime(time.localtime(time.time()))), df_indivK.shape, kapa ))
    
    df_indivK_subj = df_indivK.copy()
    df_indivK_subj['SubjID'] = 'K'
    df_indivK['SubjID']= 'K'
    num_g = (kapa-1)*num_subj

    for g in range(df_indivK.shape[0]): 
        if df_indivK.shape[0] >= 20:
            if df_indivK.shape[0]%2000==0:
                logging.info('{}: process : {}, shape : {}'.format((time.asctime(time.localtime(time.time()))), kapa, df_indivK.shape))
            df_20 = df_indivK.sample(20)
            if df_20.GameId.nunique() == 20: 
                df_indivK_subj.at[df_20.index, 'SubjID'] = num_g
                df_indivK = df_indivK.drop(df_20.index)
                df_20 = df_20[col1]
                new_row = list(np.reshape(df_20.values, (1,df_20.shape[0]*df_20.shape[1]))[0])
                new_row.extend([sigma, kapa, beta, gama, num_g])
                df_all20.loc[df_all20.shape[0]] = new_row
                num_g +=1     
            
    logging.info('{}: Finish process {} '.format((time.asctime(time.localtime(time.time()))), kapa ))
    # Split Train and Test : 
    subj_test = sample(list(df_all20.SubjID), k=int(df_all20.shape[0]*0.01))
    
    df_all20_train = df_all20.loc[~(df_all20.SubjID.isin(subj_test))]
    df_all20_test = df_all20.loc[(df_all20.SubjID.isin(subj_test))]
    df_indivK_subj_train = df_indivK_subj.loc[~(df_indivK_subj.SubjID.isin(subj_test))]
    df_indivK_subj_test = df_indivK_subj.loc[(df_indivK_subj.SubjID.isin(subj_test))]
    
    logging.info('{}: Start writting files for process {} '.format((time.asctime(time.localtime(time.time()))), kapa ))
    
    # Save DataFrame with SujbID, one game per row, train and test separatly : 
    df_indivK_subj_train.to_csv(str('/home/meghanmergui/Synth20/SynthData2/Train_' + str(kapa) +str('.csv')), index = False)
    df_indivK_subj_test.to_csv(str('/home/meghanmergui/Synth20/SynthData2/Test_' + str(kapa) +str('.csv')), index = False)

    # Save DataFrame by 20 games per row 
    df_all20_train.to_csv(str('/home/meghanmergui/Synth20/SynthData2/Train20_' + str(kapa) +str('.csv')), index = False)
    df_all20_test.to_csv(str('/home/meghanmergui/Synth20/SynthData2/Test20_' + str(kapa) +str('.csv')), index = False)
    
    logging.info('{}: Done writting files for process {} '.format((time.asctime(time.localtime(time.time()))), kapa ))

    

if __name__ == '__main__':
    
    logging.basicConfig(filename=str('MLP20_2.log'), level=logging.INFO)
    logging.info('Started')
    
    # Read Data 
    df = pd.read_csv('/home/meghanmergui/Synth20/Create_Synthetic_Games/synth_all_kapa.csv')
    logging.info('Data read')
    
    # all individual types : 
    all_types = [1,2,3]
      
    # Add PF
    PF_games = pd.read_csv('/home/meghanmergui/Synth20/Create_Synthetic_Games/2_Add_PF/Synth10KPF.csv')
    df = df.merge(PF_games, on=['Ha', 'pHa', 'La', 'LotShapeA', 'LotNumA', 'Hb', 'pHb', 'Lb', 'LotShapeB', 'LotNumB', 
                               'Corr', 'Amb'], how='left')

    del PF_games
    logging.info('PF added')
     
    # Lot Shape
    
    df['lot_shape__B'], df['lot_shape_symm_B'], df['lot_shape_L_B'], df['lot_shape_R_B'] = zip(*df['LotShapeB'].map(lot_shape_convert))
    df['lot_shape__A'], df['lot_shape_symm_A'], df['lot_shape_L_A'], df['lot_shape_R_A'] = zip(*df['LotShapeA'].map(lot_shape_convert))
    logging.info('Lotshape added')
    

    # X_col, y_col
    col1 = [ 'Ha', 'pHa', 'La', 'LotNumA', 'lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A',
             'Hb', 'pHb', 'Lb', 'LotNumB',  'lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B',
             'Corr',
             'B1',
             'Beast1',  
             'diffEV', 'diffSDs', 'diffMins', 'diffMaxs', 'diffUV', 'RatioMin', 'SignMax',
             'Dom', 'diffBEV0', 'diffSignEV', 'pBbet_Unbiased1', 'pBbet_Uniform',  'pBbet_Sign1']

# And the irrelevant features are:
    #         'Amb',
    #         'GameId',
    #         'psi', 'theta', 'wamb',
    #        'pBbet_UnbiasedFB', 'pBbet_SignFB', 'diffBEVfb'

    columns_20= []
    for i in range(1,21): 
        columns_20.extend([str(str(co) + '_'+ str(i)) for co in col1])

    x_col = columns_20
    y_col = ['sigma', 'kapa', 'beta', 'gama']
    
    num_subj = df.shape[0]/(20*3)
    
#     processes = []
    for kapa in [3]:
        df_indivK = df.loc[(df.kapa == kapa)]
        sigma = 3.5 
        beta = 1.3
        gama = 0.25
        create_MLPdata(num_subj, df_indivK, sigma, beta, kapa, gama)
#         p = multiprocessing.Process(target=create_MLPdata, args=(num_subj, df_indivK, sigma, beta, kapa, gama))
        del df_indivK
#         processes.append(p)
#         p.start()
        
#     for process in processes:
#         process.join()   
    del df
