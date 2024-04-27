from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {
            'svm': SVC(kernel='linear'),
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(),
            'decision_tree': DecisionTreeClassifier(),
            'knn': KNeighborsClassifier(),
            'gradient_boosting': GradientBoostingClassifier(),
            'ada_boost': AdaBoostClassifier(),
            'extra_trees': ExtraTreesClassifier(),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
            'naive_bayes': GaussianNB(),
            'ridge_classifier': RidgeClassifier(),
            'lda': LinearDiscriminantAnalysis(),
            'qda': QuadraticDiscriminantAnalysis()
        }

    def train(self, model_name):
        if model_name in self.models:
            model = self.models[model_name]
            model.fit(self.X_train, self.y_train)
            print(f"{model_name} model trained successfully.")
            return model
        else:
            raise ValueError(f"No model found with the name {model_name}")
    
    def split_data(self, data, target_column, test_size=0.2, random_state=None):
        if target_column not in data.columns:
            print(f"Error: Target column '{target_column}' not found in the dataset.")
            return
        X = data.drop(columns=[target_column])
        y = data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("Data split into train and test sets.")
        
    def normalize_columns(self, data, exclude_columns=None):
        if data is not None:
            scaler = StandardScaler()
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
            if exclude_columns:
                numerical_cols = [col for col in numerical_cols if col not in exclude_columns]
            if numerical_cols.any():
                data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
                print("Numerical columns normalized.")
            else:
                print("No numerical columns to normalize.")
        else:
            print("Data not provided. Please provide data before normalizing.")
        return data
