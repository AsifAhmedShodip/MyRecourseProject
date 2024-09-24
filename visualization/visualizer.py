import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import pandas as pd
import plotly.graph_objects as go
import numpy as np

class Visualizer:
    def __init__(self, model, X_test, y_test, X_original, sensitive_attr=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.X_original = X_original if X_original is not None else X_test
        self.sensitive_attr = sensitive_attr

    def plot_decision_boundary(self, feature_names):
        X = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
        y = self.y_test.values if isinstance(self.y_test, pd.Series) else self.y_test

        plt.figure(figsize=(10, 6))
        # Plot decision regions using mlxtend's plot_decision_regions function
        plot_decision_regions(X, y.astype(int), clf=self.model, legend=2)
        
        # Customizing the plot
        plt.title('Decision Boundary with Test Data')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.legend(loc='upper left')
        plt.show()

    def plot_decision_boundary_interactive(self, feature_names, plot_title):
        x_min, x_max = self.X_test[:, 0].min() - 1, self.X_test[:, 0].max() + 1
        y_min, y_max = self.X_test[:, 1].min() - 1, self.X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig = go.Figure()

        # Use Heatmap to set different colors for different regions
        fig.add_trace(go.Heatmap(
            x=np.linspace(x_min, x_max, 500),
            y=np.linspace(y_min, y_max, 500),
            z=Z,
            colorscale=[[0, 'red'], [1, 'blue']],
            showscale=False,
            opacity=0.4
        ))

        

        # Scatter plot for data points
        marker_colors = np.where(self.y_test == 1, 'blue', 'red')  # blue for positive, red for negative
        marker_lines = np.where(self.sensitive_attr == 2, 'white', 'black')  # white ring for females

        fig.add_trace(go.Scatter(
            x=self.X_test[:, 0], y=self.X_test[:, 1], mode='markers',
            marker=dict(
                color=marker_colors,  # Use class labels to determine inner color
                size=10,
                line=dict(width=2, color=marker_lines)  # Use gender to set ring color
            ),
            customdata=self.X_original,
            hovertemplate="<br>".join([
                f"{feature_names[0]}: %{{customdata[0]}}",
                f"{feature_names[1]}: %{{customdata[1]}}",
                # "Class: %{{marker.color}}"
                f"Gender: %{{text}}"
            ]),
            text=['Male' if x == 1 else 'Female' for x in self.sensitive_attr]
        ))

        annotations = [
            dict(
                x=1, y=1.0, xref='paper', yref='paper',
                text='Class 0: Red<br>Class 1: Blue<br>Male: Black Ring<br>Female: White Ring',
                showarrow=False,
                align='left',
                font=dict(size=14, color='black', family='Arial, bold')  # Adjust font size and make it bold
            )
    ]


        fig.update_layout(
            title=plot_title,
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1],
            annotations=annotations,
            legend_title="Class and Gender",
            legend=dict(
                x=1,
                y=1,
                bordercolor="Black",
                borderwidth=2
            )
        )

        fig.show()