from functions import add_bias_weights,find_best_probs,MSE_biases
import pandas as pd
if __name__ == '__main__':
    df=pd.read_csv('c13k_selections_weights.csv')
    df['Pred_By_Prob_MSE']=MSE_biases([3.74624893e-01,3.24310085e-01,2.60208521e-18,3.01065022e-01,
                     5.24355420e-01,2.36038734e-01,4.33680869e-18,2.39605846e-01,
                     5.59580517e-01,2.16810566e-01,0.00000000e+00,2.23608916e-01,
                     5.81619469e-01,2.04661293e-01,0.00000000e+00,2.13719237e-01,
                     5.87916457e-01,1.87924074e-01,2.28747609e-02,2.01284709e-01],df, 'bRate')
    df.to_csv('c13k_selections_weights.csv',index=False)