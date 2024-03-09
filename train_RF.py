from matplotlib.lines import Line2D
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def main():
    data_dir='phreeqc_data_SCM_params_DNN/original_data'

    train = np.load('dataset/train_compress.npz')
    test = np.load('dataset/test_compress.npz')
    
    x_train_2pk = train['x']
    y_train_2pk = train['y']
    
    x_test_2pk = test['x']
    y_test_2pk = test['y']
    
    import time 
    start_time = time.perf_counter()
    estimator_RF = RandomForestRegressor(n_estimators=100,oob_score=True, random_state=10, n_jobs=-1)
    estimator_RF.fit(x_train_2pk, y_train_2pk)
    end_time = time.perf_counter() 

    print("training time is: ", (end_time - start_time)/60, 'mins')

    RF_pred = estimator_RF.predict(x_test_2pk)
    np.savetxt('pred_RF.txt', RF_pred)

    # save the model to disk
    import pickle
    filename = 'RF.sav'
    pickle.dump(estimator_RF, open(filename, 'wb'))
    

if __name__ == '__main__':
    main()
