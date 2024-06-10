from data_handling.data_handler import DataHandler

class Dataset(DataHandler):
    def __init__(self, target_column=None):
        super().__init__(target_column=target_column)

    def load_csv(self, file_path=None):
        if file_path is None:
            file_path = '/Users/asifahmed/Documents/Codes/MyRecourseProject/datasets/credit_processed.csv'

        super().load_csv(file_path)

    def preprocess(self):
        # Implement any specific preprocessing needed for the Bank dataset
        # Example: Encoding categorical variables, handling missing values, etc.
        self.encode_categorical_columns()
        # self.fill_missing_values(strategy='median')
