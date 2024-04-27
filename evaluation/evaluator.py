from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np

class Evaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None

    def evaluate(self):
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        
    def get_metric(self, metric):
        self.evaluate()
        return metric(self.y_test, self.y_pred)

    def report(self):
        self.evaluate()
        print("Accuracy:", self.get_metric(accuracy_score))
        print("Precision:", self.get_metric(precision_score))
        print("Recall:", self.get_metric(recall_score))
        print("F1 Score:", self.get_metric(f1_score))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, self.y_pred))
        print("Classification Report:\n", classification_report(self.y_test, self.y_pred))

    def find_misclassified_samples(self):
        self.evaluate()
        return np.where(self.y_pred != self.y_test)[0]
    
    def find_correctly_classified_samples(self):
        self.evaluate()
        return np.where(self.y_pred == self.y_test)[0]
