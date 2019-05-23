import pymongo
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#  from statsmodels.api import OLS
#  import statsmodels.api as sm
#  import shap
#  import matplotlib.pyplot as plt
import pickle
load_dotenv()


class Model:
    mapping = {'Andrej': 0, 'Felipe': 1, 'Gayane': 2, 'Igor': 3,
               'Michaela': 4, 'Pavol': 5, 'Petra': 6, 'Tolja': 7}

    def __init__(self):
        self.model = self.load_model()

    def predict(self, p1, p2, p3, p4):
        message_array = [0]*8
        message_array[self.mapping[p1]] = 1
        message_array[self.mapping[p2]] = 1
        message_array[self.mapping[p3]] = -1
        message_array[self.mapping[p4]] = -1
        #  message = json.dumps([message_array])
        return self._predict([message_array])

    def _predict(self, dataset):
        predictions = self.model.predict_proba(dataset)
        predictions = [pred[1] for pred in predictions]
        return predictions

    def retrain(self):
        df_records, unplayed = get_df_records()
        X = transform_df(df_records)
        # X_test = transform_df(unplayed)
        y = df_records[4]
        X_train, X_dev, y_train, y_dev = train_test_split(
            X, y, test_size=0.2, random_state=41)

        self.model = LogisticRegression(
            random_state=42,
            solver='liblinear',
            verbose=1
        ).fit(X_train, y_train)

        self.model.coef_
        print(classification_report(y_dev, self.model.predict(X_dev)))
        self.save_model(self.model)

    @staticmethod
    def save_model(model, filename='model/model.pkl'):
        pickle.dump(model, open(filename, 'wb'))

    @staticmethod
    def load_model(filename='model/model.pkl'):
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model


def transform_df(df_records):
    t1 = pd.get_dummies(df_records[0]).add(
        pd.get_dummies(df_records[1]), fill_value=0)
    t2 = pd.get_dummies(df_records[2]).add(
        pd.get_dummies(df_records[3]), fill_value=0)
    df = t1.add(t2*-1, fill_value=0)
    return df


def get_df_records():
    client = pymongo.MongoClient(os.environ['DB_URI'])
    db = client.test
    records = db.matches
    records.count_documents({})
    all_records = list(records.find({}))
    all_records_list = [
        [
            record['team1'][0]['name'],
            record['team1'][1]['name'],
            record['team2'][0]['name'],
            record['team2'][1]['name'],
            record['score']
        ] for record in all_records
    ]
    df_records = pd.DataFrame(all_records_list)
    unplayed = df_records[df_records[4] == '0-0']
    df_records = df_records[df_records[4] != '0-0']
    df_records[4] = df_records[4].apply(lambda x: 1 * (x[0] == '2'))
    return df_records, unplayed
