import numpy as np
import pandas as pd
import pickle
import yaml

# read yaml file
with open("config.yaml") as file:
    config = yaml.safe_load(file)


model = pd.read_pickle("randomforest.pkl")


def make_prediction(input_data):
    data = pd.DataFrame(input_data)
    prediction = model.predict(data[config["FEATURES"]])
    result = pd.concat([data[config["FEATURES"]], pd.DataFrame(prediction)], axis=1)
    result = result.rename(columns={0: "Prediction"})
    result.to_csv("out.csv")
