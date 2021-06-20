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


def create_MLPdata(df, type_3, i): 
    df_all20 = pd.DataFrame(columns=x_col+y_col)
    logging.info('{} : process {} started'.format((time.asctime(time.localtime(time.time()))), i))
    for indiv in type_3:
        
        sigma = indiv[0]
        beta = indiv[1]
        kapa =indiv[2]
        gama = indiv[3]

        df_indiv = df.loc[(df.sigma == sigma) & (df.beta == beta) & (df.kapa == kapa) & (df.gama == gama)]
        del df
        logging.info('{}: Start indiv : {}, shape : {}, process {}'.format((time.asctime(time.localtime(time.time()))), indiv, df_indiv.shape, i ))

        num_g = 0
        while df_indiv.shape[0] >= 20 : 
            if df_indiv.shape[0] % 200 : 
                logging.info('{}: process : {}, shape : {}'.format((time.asctime(time.localtime(time.time()))), i, df_indiv.shape))
            df_20 = df_indiv.sample(20)
            if df_20.GameId.nunique() == 20 : 
                num_g +=1
                df_indiv = df_indiv.drop(df_20.index)
                df_20 = df_20[col1]
                new_row = list(np.reshape(df_20.values, (1,df_20.shape[0]*df_20.shape[1]))[0])
                new_row.extend([sigma, kapa, beta, gama])
                df_all20.loc[df_all20.shape[0]] = new_row
#    print(i)
#    print(df_all20.shape)
#    print(df_all20.head(5))

    df_all20.to_csv(str('/home/meghanmergui/Synth20/Create_Synthetic_Games/Synth20MLP/synth_20MLP' + str(i) +str('.csv')), index = False)

    

if __name__ == '__main__':
    
    logging.basicConfig(filename=str('MLPcreate20_.log'), level=logging.INFO)
    logging.info('Started')
    
    # Read Data 
    df = pd.read_csv('/home/meghanmergui/Synth20/Create_Synthetic_Games/synth_all.csv')
    logging.info('Data read')
    
    # all individual types : 
    all_types = [(1.17, 0.43, 1, 0.33), (1.17, 0.43, 1, 1.5), (1.17, 0.43, 2, 0.33), (1.17, 0.43, 2, 1.5), (1.17, 0.43, 3, 0.33), (1.17, 0.43, 3, 1.5), (1.17, 1.3, 1, 0.33), (1.17, 1.3, 1, 1.5), (1.17, 1.3, 2, 0.33), (1.17, 1.3, 2, 1.5), (1.17, 1.3, 3, 0.33), (1.17, 1.3, 3, 1.5), (1.17, 2.17, 1, 0.33), (1.17, 2.17, 1, 1.5), (1.17, 2.17, 2, 0.33), (1.17, 2.17, 2, 1.5), (1.17, 2.17, 3, 0.33), (1.17, 2.17, 3, 1.5), (3.5, 0.43, 1, 0.33), (3.5, 0.43, 1, 1.5), (3.5, 0.43, 2, 0.33), (3.5, 0.43, 2, 1.5), (3.5, 0.43, 3, 0.33), (3.5, 0.43, 3, 1.5), (3.5, 1.3, 1, 0.33), (3.5, 1.3, 1, 1.5), (3.5, 1.3, 2, 0.33), (3.5, 1.3, 2, 1.5), (3.5, 1.3, 3, 0.33), (3.5, 1.3, 3, 1.5), (3.5, 2.17, 1, 0.33), (3.5, 2.17, 1, 1.5), (3.5, 2.17, 2, 0.33), (3.5, 2.17, 2, 1.5), (3.5, 2.17, 3, 0.33), (3.5, 2.17, 3, 1.5), (5.83, 0.43, 1, 0.33), (5.83, 0.43, 1, 1.5), (5.83, 0.43, 2, 0.33), (5.83, 0.43, 2, 1.5), (5.83, 0.43, 3, 0.33), (5.83, 0.43, 3, 1.5), (5.83, 1.3, 1, 0.33), (5.83, 1.3, 1, 1.5), (5.83, 1.3, 2, 0.33), (5.83, 1.3, 2, 1.5), (5.83, 1.3, 3, 0.33), (5.83, 1.3, 3, 1.5), (5.83, 2.17, 1, 0.33), (5.83, 2.17, 1, 1.5), (5.83, 2.17, 2, 0.33), (5.83, 2.17, 2, 1.5), (5.83, 2.17, 3, 0.33), (5.83, 2.17, 3, 1.5)]

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
             'Dom', 'diffBEV0', 'diffBEVfb', 'diffSignEV']
    #         'Amb',
    #         'GameId',
    #         'psi', 'theta', 'wamb',
    #        'pBbet_Unbiased1', 'pBbet_UnbiasedFB', 'pBbet_Uniform', 'pBbet_Sign1', 'pBbet_SignFB',

    columns_20= []
    for i in range(1,21): 
        columns_20.extend([str(str(co) + '_'+ str(i)) for co in col1])

    x_col = columns_20
    y_col = ['sigma', 'kapa', 'beta', 'gama']
    
    processes = []
    
    for i in range(0,6):
        type_3i = all_types[i*3:(i+1)*3]
        create_MLPdata(df, type_3i, i)
#        p = multiprocessing.Process(target=create_MLPdata, args=(df, type_3i, i))
#        processes.append(p)
     #   p.start()

#    for process in processes:
#        process.join()    


