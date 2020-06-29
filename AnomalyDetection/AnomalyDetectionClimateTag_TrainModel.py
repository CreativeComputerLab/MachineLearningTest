import pandas as pd 
import numpy as np 
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

if __name__ == '__main__':
    #storage 
    data_file = 'climate sensor tag_data - noDates - flat.csv' 
    model_file = "model.sav"
    scaler_file = "scaler.sav"

    #classification results from training data
    anomaly_count = 0
    normal_count = 0

    #get data
    df = pd.read_csv(data_file)   

    #convert type for sklearn
    training_data = np.asarray(df)
    for t in training_data:
        print(t)
        
    # TODO - Data pre-processing step 
    # Either eliminate the date column altogether or convert it to unix epoch time.

    #standardize - this normalizes between 1 and 1. This data and can be used for plotting simultaneous lines
    scaler = MinMaxScaler().fit(training_data)
    training_data_transformed = scaler.transform(training_data)
    for t in training_data_transformed:
        print(t)
 
    #create model
    model = IsolationForest()
    model.fit(training_data_transformed)
    prediction = model.predict(training_data_transformed)

    #see classification results 
    for p in prediction:
        if p == -1:            
            anomaly_count += 1
            print("anomaly ", p)
        else:
            normal_count += 1  
            print("normal ", p)  
    print("anomaly count: ", anomaly_count)
    print("normal count: ", normal_count)

    #save model & scaler for later application
    pickle.dump(model, open(model_file, 'wb'))
    pickle.dump(scaler, open(scaler_file, 'wb'))

