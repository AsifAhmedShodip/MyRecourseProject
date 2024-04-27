from data_handling.data_handler import DataHandler

class GiveMeCredit(DataHandler):
    def __init__(self):
        super().__init__()

    def load_csv(self, file_path=None):
        if file_path is None:
            file_path = 'datasets/GivemeTraining_Original.csv'

        super().load_csv(file_path)

    def preprocess(self):
        # Implement any specific preprocessing needed for the Bank dataset
        # Example: Encoding categorical variables, handling missing values, etc.
        self.encode_categorical_columns()
        # self.fill_missing_values(strategy='median')
