import numpy as np
import pandas as pd

class RecourseCostCalculator:
    def __init__(self, model):
        self.model = model
        self.weights = model.coef_[0]
        self.intercept = model.intercept_[0]

    def calculate_distance(self, data_point):
        numerator = np.abs(np.dot(self.weights, data_point) + self.intercept)
        denominator = np.linalg.norm(self.weights)
        return numerator / denominator

    def calculate_recourse_costs(self, data):
        return np.array([self.calculate_distance(point) for point in data])

    def calculate_group_recourse_costs(self, data, group_identifiers):
        # Ensure that the data is in numpy array format if it's a pandas DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        # Similarly, ensure that the group identifiers are in numpy array format
        if isinstance(group_identifiers, (pd.Series, pd.DataFrame)):
            group_identifiers = group_identifiers.to_numpy()

        group_costs = {}
        unique_groups = np.unique(group_identifiers)

        for group in unique_groups:
            group_mask = group_identifiers == group
            group_data = data[group_mask]
            costs = self.calculate_recourse_costs(group_data)
            group_costs[group] = costs

        return group_costs