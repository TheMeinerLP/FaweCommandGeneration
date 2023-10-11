import sys

import pandas as pd
from sklearn.model_selection import train_test_split


def generate_data():
    if len(sys.argv) == 1:
        prefix = sys.argv[0]
        data = pd.read_csv(F'../../datasets/{prefix}_train.csv')
        train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
        valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        valid_data.to_csv(F'../../datasets/{prefix}_valid.csv', index=False)
        test_data.to_csv(F'../../datasets/{prefix}_test.csv', index=False)


sys.argv = ['blocks']
generate_data()
