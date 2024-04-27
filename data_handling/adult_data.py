from data_handling.data_handler import DataHandler
from sklearn.preprocessing import LabelEncoder


class Adult(DataHandler):
    def __init__(self):
        super().__init__()

    def load_csv(self, file_path=None):
        if file_path is None:
            file_path = 'datasets/adult.csv'

        super().load_csv(file_path)

    def preprocess(self):
        # Adult specific preprocessing steps
        # self.fill_missing_values(strategy='mode')
        self.encode_categorical_columns()