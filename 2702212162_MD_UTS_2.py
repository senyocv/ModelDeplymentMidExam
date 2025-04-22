import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None
        self.x_train = self.x_test = self.y_train = self.y_test = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data.dropna(inplace=True)

    def create_input_output(self, input_columns, target_column):
        self.input_df = self.data[input_columns]
    
        le = LabelEncoder()
        self.output_df = pd.DataFrame(le.fit_transform(self.data[target_column]), columns=[target_column])
        self.label_encoder = le

    def split_data(self, test_size=0.2, random_state=354):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_df, self.output_df, test_size=test_size, random_state=random_state)
        return self.x_train, self.x_test, self.y_train, self.y_test


class EncoderHandler:
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

    def fit_transform(self, df):
        df = df.copy()
        enc_arr = self.ohe.fit_transform(df[self.categorical_columns])
        enc_df = pd.DataFrame(enc_arr, columns=self.ohe.get_feature_names_out(self.categorical_columns))
        
        df = df.drop(columns=self.categorical_columns).reset_index(drop=True)
        enc_df = enc_df.reset_index(drop=True)
        df = pd.concat([df, enc_df], axis=1)
        
        return df

    def transform(self, df):
        df = df.copy()
        enc_arr = self.ohe.transform(df[self.categorical_columns])
        enc_df = pd.DataFrame(enc_arr, columns=self.ohe.get_feature_names_out(self.categorical_columns))
        
        df = df.drop(columns=self.categorical_columns).reset_index(drop=True)
        enc_df = enc_df.reset_index(drop=True)
        df = pd.concat([df, enc_df], axis=1)
        
        return df


class ModelHandler:
    def __init__(self):
        self.param_grid = {
            'n_estimators': [100],
            'max_depth': [10],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt'],
            'bootstrap': [True],
            'criterion': ['gini']
        }
        self.model = GridSearchCV(RandomForestClassifier(), self.param_grid, cv=3, n_jobs=-1, scoring='accuracy')

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        print("Best Parameters:", self.model.best_params_)

    def evaluate(self, x_test, y_test):
        preds = self.model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        print("\nAccuracy: {:.5f}".format(acc))
        print("\nClassification Report:\n", classification_report(y_test, preds, digits=3))

    def save(self, filename="rf_model.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.model.best_estimator_, f)
        print(f"Model saved to {filename}")



file_path = "Dataset_B_hotel.csv"
input_col = [col for col in pd.read_csv(file_path).columns if col != "booking_status"]
target_col = "booking_status"
cat_col = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']

data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output(input_col, target_col)
x_train, x_test, y_train, y_test = data_handler.split_data()

encoder = EncoderHandler(categorical_columns=cat_col)
x_train_enc = encoder.fit_transform(x_train)
x_test_enc = encoder.transform(x_test)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train.values.ravel())
y_test_enc = le.transform(y_test.values.ravel())

model_handler = ModelHandler()
model_handler.train(x_train_enc, y_train_enc)
model_handler.evaluate(x_test_enc, y_test_enc)
model_handler.save("rf_md_uts.pkl")



















