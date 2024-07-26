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
        distance = numerator / denominator
        return round(distance, 2)

    def calculate_recourse_costs(self, data):
        return np.array([self.calculate_distance(point) for point in data])
        

    def calculate_group_recourse_costs(self, data, group_identifiers):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
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
    
    def rank_data_points(self, scaled_data, gender_data, top_n=None):
        costs = self.calculate_recourse_costs(scaled_data)
        sorted_indices = np.argsort(costs)

        if top_n is None:
            top_n = len(costs)

        top_indices = sorted_indices[:top_n]
        top_costs = costs[top_indices]

        data = pd.DataFrame(scaled_data[top_indices], columns=['Feature1', 'Feature2'])
        data['Index'] = top_indices  
        data['Gender'] = gender_data.iloc[top_indices].values
        data['Rank'] = range(1, top_n + 1)
        data['Recourse Cost'] = top_costs

        data = data[['Index', 'Feature1', 'Feature2', 'Gender', 'Rank', 'Recourse Cost']]

        return data
    
    def apply_gender_rewards(self, data):
        # Make a copy to avoid changes to the original DataFrame outside this function
        data_modified = data.copy()

        # Store original ranks
        data_modified['Previous Rank'] = data_modified['Rank'].copy()

        # Apply min-max scaling to female recourse costs
        female_costs = data_modified[data_modified['Gender'] == 2]['Recourse Cost']
        min_cost = female_costs.min()
        max_cost = female_costs.max()
        scaling_factor = (female_costs - min_cost) / (max_cost - min_cost) * 0.9 + 0.1
        data_modified.loc[data_modified['Gender'] == 2, 'Recourse Cost'] = scaling_factor.round(2)

        # Sort data by updated recourse costs and reassign ranks
        data_modified.sort_values(by='Recourse Cost', ascending=True, inplace=True)
        data_modified.reset_index(drop=True, inplace=True)
        data_modified['Rank'] = range(1, len(data_modified) + 1)

        # Specify the column order to include 'Previous Rank' before 'Rank'
        columns_order = ['Index', 'Feature1', 'Feature2', 'Gender', 'Previous Rank', 'Rank', 'Recourse Cost']
        data_modified = data_modified[columns_order]
        
        return data_modified
    

    def identify_females_to_help(self, recourse, percentage_to_help):
            # Filter to get only female data points
            female_data = recourse[recourse['Gender'] == 2]
            # Calculate the cutoff for the number of females to help
            cutoff = int(len(female_data) * percentage_to_help)
            # Select the females who are farthest from the decision boundary
            females_to_help = female_data.nlargest(cutoff, 'Recourse Cost')
            return females_to_help
    
    def calculate_gradient_based_recourse(self, data):
        """ Calculate gradient-based recourse for each sample in data. """
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        gradient = self.weights
        norm_gradient = np.linalg.norm(gradient)
        inversed_decision_function = -(np.dot(data, gradient) + self.intercept)

        # Calculate minimal perturbation to flip the decision
        minimal_perturbation = np.abs(inversed_decision_function) / norm_gradient
        return minimal_perturbation
    
    def calculate_feature_perturbation_costs(self, data):
        base_predictions = self.model.predict(data)
        costs = np.zeros(data.shape)
        for i in range(data.shape[1]):  # Iterate over each feature
            perturbed_data = data.copy()
            perturbed_data[:, i] += 0.01  # Small perturbation
            perturbed_predictions = self.model.predict(perturbed_data)
            change_mask = perturbed_predictions != base_predictions
            costs[change_mask, i] = 0.01
        return np.sum(costs, axis=1)
    
    from scipy.optimize import minimize

    def calculate_counterfactual_distance(self, data):
        """ Calculate the perpendicular distance from the decision boundary for each sample. """
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        # Calculate the distance from the decision boundary
        distance = np.abs(np.dot(data, self.weights) + self.intercept) / np.linalg.norm(self.weights)
        return distance
    

    def calculate_sensitivity_costs(self, data):
        if hasattr(self.model, 'predict_proba'):
            base_probabilities = self.model.predict_proba(data)
            sensitivity = np.zeros(data.shape[0])
            for i in range(data.shape[1]):
                perturbed_data = data.copy()
                perturbed_data[:, i] += 0.01
                perturbed_probabilities = self.model.predict_proba(perturbed_data)
                sensitivity += np.abs(base_probabilities - perturbed_probabilities).max(axis=1)
            return sensitivity
        else:
            raise AttributeError("Model does not support probability estimation.")

            