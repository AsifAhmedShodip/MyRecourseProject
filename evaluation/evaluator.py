from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

class Evaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None

    def evaluate(self):
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        
    def get_metric(self, metric, **kwargs):
        self.evaluate()
        return metric(self.y_test, self.y_pred, **kwargs)

    def report(self):
        self.evaluate()
        print("Accuracy:", self.get_metric(accuracy_score))
        print("Precision:", self.get_metric(precision_score, zero_division=0))
        print("Recall:", self.get_metric(recall_score, zero_division=0))
        print("F1 Score:", self.get_metric(f1_score, zero_division=0))
        print("Confusion Matrix:\n", self.get_metric(confusion_matrix))
        print("Classification Report:\n", classification_report(self.y_test, self.y_pred, zero_division=0))

    def get_evaluation_metrics(self):
        self.evaluate()
        metrics = {
            'Accuracy': self.get_metric(accuracy_score),
            'Precision': self.get_metric(precision_score, zero_division=0),
            'Recall': self.get_metric(recall_score, zero_division=0),
            'F1 Score': self.get_metric(f1_score, zero_division=0),
            'Confusion Matrix': self.get_metric(confusion_matrix),
            'Classification Report': classification_report(self.y_test, self.y_pred, zero_division=0)
        }
        return metrics