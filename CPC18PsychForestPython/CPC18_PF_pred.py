from get_PF_Features import get_PF_Features
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def CPC18_PF_pred(train_data, Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
    # Prediction of Psychological Forest for one problem
    #
    #  This function gets as input 12 parameters which define a problem in CPC18
    #  and outputs Psych. Forest's prediction in that problem for five blocks of
    #  five trials each (the first is without and the others are with feedback

    # get data frame of engineered features's values for the prediction problem
    # Output: the prediction of 5 iteration of the game (numpy array of shape(1,5))
    Feats = get_PF_Features(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)

    n_runs = 10
    prediction = np.repeat([0], 5)
    prediction.shape = (1, 5)
    for run in range(n_runs):
        # train a random forest algorithm using all supplied features of the train data
        x_train = train_data.iloc[:, 1:38]
        y_train = train_data['B_rate']
        rf_model = RandomForestRegressor(n_estimators=500, max_features=0.3333, min_samples_leaf=5)
        rf_model.fit(X=x_train, y=y_train)

        # let the trained RF predict the prediction prbolem
        pred = rf_model.predict(Feats)
        prediction = np.add(prediction, (1 / n_runs) * pred)

    return prediction
