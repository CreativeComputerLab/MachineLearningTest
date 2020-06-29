import pandas as pd 
import numpy as np
import pickle

'''
get_data    placeholder for collecting event data from Modi device
'''
def get_data():
    df = pd.read_csv("climate sensor tag_data - noDates - lightSpike.csv")
    training_data = np.asarray(df)
    return training_data


'''
classify    Classifies data as anomaly or normal behavior 

@param model    trained model to classify event data
@param scaler   MinMaxScaler used to scale training data for the model
@param ANOMALY  Constant value signaling anomaly from model prediction 
'''
def classify(model, scaler, ANOMALY):
    i = 0
    events = get_data()
    transformed_data = scaler.transform(events)
    classification = model.predict(transformed_data)

    #print event data with labels 
    for p in classification:
        if p == ANOMALY:
            #TODO send data to data visualization service 
            print("Anomaly: ", events[i]) #placeholder
        else:
            print("Normal: ", events[i]) #placeholder
        i += 1


if __name__ == '__main__':
    ANOMALY = -1
    model_file = "model.sav"
    scaler_file = "scaler.sav"

    #load model & scaler
    model = pickle.load(open(model_file, 'rb'))
    scaler = pickle.load(open(scaler_file, 'rb'))

    classify(model, scaler, ANOMALY)
