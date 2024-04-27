# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# class Visualizer:
#     def __init__(self):
#         pass

#     def plot_decision_boundary(self, model, X, y, misclassified_indices):
#         h = .02  # step size in the mesh
#         x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#         y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#         # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#         # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

#         # Z = Z.reshape(xx.shape)
#         # plt.figure(figsize=(8, 6))
#         # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

#         # plt.scatter(X[misclassified_indices, 0], X[misclassified_indices, 1],
#         #             c='black', marker='x', label='Misclassified')

#         # # plt.xlabel(selected_columns[0])
#         # # plt.ylabel(selected_columns[1])
#         # plt.title('Decision Boundary')
#         # plt.show()

#     def plot_feature_distributions(self, X, features):
#         for column_index, feature_name in enumerate(features):
#             plt.figure(figsize=(8, 6))
#             sns.histplot(data=X[:, column_index], kde=True)
#             plt.title(f'Distribution of {feature_name}')
#             plt.xlabel(feature_name)
#             plt.ylabel('Frequency')
#             plt.show()


# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import numpy as np

# class Visualizer:
#     def __init__(self, model, X_test, y_test, y_pred, features):
#         self.model = model
#         self.X_test = X_test[features]  # Ensure only the relevant features are used
#         self.y_test = y_test
#         self.y_pred = y_pred
#         self.features = features

#     def plot_decision_boundary(self):
#         h = 0.01  # Increase resolution of the mesh grid
#         x_min, x_max = self.X_test.iloc[:, 0].min() - 1, self.X_test.iloc[:, 0].max() + 1
#         y_min, y_max = self.X_test.iloc[:, 1].min() - 1, self.X_test.iloc[:, 1].max() + 1
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#         Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])


#         # # Predict decision function or probabilities
#         # if hasattr(self.model, "decision_function"):
#         #     Z = self.model.decision_function(np.c_[xx.ravel(), yy.ravel()])
#         #     levels = [0]
#         #     linestyles = ['dashed']
#         #     colors = 'black'
#         # else:
#         #     Z = self.model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#         #     levels = [0.5]  # Change level for probability-based models
#         #     linestyles = ['dashed']
#         #     colors = 'red'  # Make the boundary more visible

#         Z = Z.reshape(xx.shape)

#         # Plot the contour and training examples
#         plt.figure(figsize=(10, 6))
#         plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
#         plt.scatter(self.X_test.iloc[:, 0], self.X_test.iloc[:, 1], c=self.y_test, cmap=plt.cm.coolwarm, edgecolor='k')

#         plt.title("Decision Boundary with Test Data")
#         plt.xlabel(self.features[0])
#         plt.ylabel(self.features[1])
#         plt.show()

#         # Debugging: Check min and max of Z to ensure correct levels are used
#         print("Min of Z:", Z.min())
#         print("Max of Z:", Z.max())


import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import pandas as pd

class Visualizer:
    def __init__(self, model, X_test, y_test):
        """
        Initializes the Visualizer with a trained model and test dataset.
        
        :param model: A trained scikit-learn compatible model, such as an SVM.
        :param X_test: Test features (numpy array) - should be scaled if the model requires it.
        :param y_test: True labels for test data (numpy array).
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def plot_decision_boundary(self):
        X = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
        y = self.y_test.values if isinstance(self.y_test, pd.Series) else self.y_test

        plt.figure(figsize=(10, 6))
        # Plot decision regions using mlxtend's plot_decision_regions function
        plot_decision_regions(X, y.astype(int), clf=self.model, legend=2)
        
        # Customizing the plot
        plt.title('Decision Boundary with Test Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(loc='upper left')
        plt.show()

