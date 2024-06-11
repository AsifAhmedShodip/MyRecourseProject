from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None  # Initialize a model attribute
        self.models = {
            'svm': SVC(kernel='linear'),
            'svm_poly': SVC(kernel='poly', degree=3),
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

    def split_data(self, data, target_column, test_size=0.2, random_state=None):
        if target_column not in data.columns:
            print(f"Error: Target column '{target_column}' not found in the dataset.")
            return None, None, None, None  # Return a tuple of None to avoid errors when unpacking
        X = data.drop(columns=[target_column])
        y = data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("Data split into train and test sets.")
        return self.X_train, self.X_test, self.y_train, self.y_test  # Return the split data

    def scale_features(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train(self, model_name, X_train=None, y_train=None):
        # Use internal data if no data is provided
        if X_train is None or y_train is None:
            X_train = self.X_train
            y_train = self.y_train
        if model_name in self.models:
            self.model = self.models[model_name]  # Store the current model after training
            self.model.fit(X_train, y_train)
            print(f"{model_name} model trained successfully.")
            return self.model
        else:
            raise ValueError(f"No model found with the name {model_name}")

    def get_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"No model found with the name {model_name}")
        

    def find_misclassified_samples(self):
        y_pred = self.model.predict(self.X_test)
        misclassified = self.y_test != self.y_pred
        return np.where(misclassified)[0]  # Indices of misclassified samples
        

    def distance_to_hyperplane(self, point):
        """
        Calculate the distance from a point to the decision boundary of the trained linear model.
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("This method is only applicable to linear models.")
        coef = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        num = np.abs(np.dot(coef, point) + intercept)
        denom = np.linalg.norm(coef)
        return num / denom