import os

import pandas as pd
HOUSING_PATH=r"C:\Users\Administrator\Desktop\Learning machine\housing"
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)
