import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
import numpy as np



class DataHandler:
    def __init__(self, target_column=None):
        self.data = None
        self.group = None
        self.target_column = target_column

    def load_csv(self, file_path, delimiter=None):
        # Initialize delimiter early to ensure it's always defined
        if delimiter is None:
            delimiter = ','  # Set a default delimiter

        try:
            # Open the file to detect the delimiter if it wasn't explicitly provided
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                header = csvfile.read(2048)
                csvfile.seek(0)  # Reset to the start of the file
                dialect = csv.Sniffer().sniff(header, delimiters=',;\t| ')
                delimiter = dialect.delimiter  # Update delimiter based on detection

            # Load the data using pandas with the detected or provided delimiter
            self.data = pd.read_csv(file_path, delimiter=delimiter)
            print(f"CSV file loaded successfully with delimiter: '{delimiter}'")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            self.data

    def group_data(self, column_name):
        if self.data is not None:
            if column_name in self.data.columns:
                self.group = self.data.groupby(column_name)
                print(f"Data has been grouped by {column_name}.")
            else:
                print(f"Column '{column_name}' not found in the data.")
        else:
            print("Data not loaded. Please load data before trying to group it.")

    def print_groups(self):
        """
        Prints each group created by the group_data method.
        """
        if self.group is not None:
            for name, group in self.group:
                print(f"Group by '{name}':")
                print(group)
                print("\n")  # Adds a newline for better readability between groups
        else:
            print("No groups to display. Please group data first.")

    def get_feature_names(self):
        if self.data is not None:
            return self.data.columns.tolist()
        else:
            return "Data not loaded"
        
    def print_features_names(self):
        """ Prints the names of the features in the dataset. """
        if self.data is not None:
            feature_names = self.data.columns.tolist()
            print("Feature names in the dataset:")
            for name in feature_names:
                print(name)
        else:
            print("Data not loaded. Please load data to see feature names.")

    def select_features(self, features):
        if self.data is not None:
            missing_features = [f for f in features if f not in self.data.columns]
            if missing_features:
                print("Warning: The following features are not in the dataset and will be ignored:", missing_features)
            valid_features = [f for f in features if f in self.data.columns]
            self.data = self.data[valid_features]
            print("Selected features are now active:", valid_features)
        else:
            print("Data not loaded. Please load data before selecting features.")

    def encode_categorical_columns(self):
        if self.data is not None:
            label_encoder = LabelEncoder()
            for column in self.data.columns:
                if self.data[column].dtype == 'object':
                    self.data[column] = label_encoder.fit_transform(self.data[column])
                    # print(f"Encoded {column}")
        else:
            print("Data not loaded. Please load data before encoding.")


    def sample_data(self, sample_size=2000):
        if self.data is not None:
            # Ensure that the sample size is not greater than the number of rows in the dataset
            actual_size = min(sample_size, len(self.data))
            # Randomly select 'actual_size' rows from the data and update self.data
            self.data = self.data.sample(n=actual_size, random_state=42)  # random_state for reproducibility
            print(f"Data has been sampled. New data size: {len(self.data)} rows.")
        else:
            print("Data not loaded. Please load data before sampling.")

    def group_by_column(self, column_name):
        if self.data is not None:
            if column_name in self.data.columns:
                self.group = self.data.groupby(column_name)
                print(f"Data has been grouped by {column_name}.")
            else:
                print(f"Column '{column_name}' not found in the data.")
                self.group = None
        else:
            print("Data not loaded. Please load data before grouping.")
            self.group = None

    def preview_data(self, rows=5):
        return self.data.head(rows)

    def print_summary_statistics(self):
        """ Prints summary statistics of the dataset. """
        if self.data is not None:
            print("Summary Statistics:")
            print(self.data.describe().to_string())  # to_string() to print full summary
        else:
            print("Data not loaded. Please load data to view summary statistics.")

    def check_missing_values(self):
        return self.data.isnull().sum()

    def drop_missing_values(self):
        self.data.dropna(inplace=True)
        print("Missing values dropped.")

    def fill_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.columns
        
        for column in columns:
            if self.data[column].dtype == 'object' and strategy in ['mean', 'median', 'constant']:
                fill_value = self.data[column].mode()[0] if strategy != 'constant' else 'missing'
            elif strategy == 'mean':
                fill_value = self.data[column].mean()
            elif strategy == 'median':
                fill_value = self.data[column].median()
            elif strategy == 'mode':
                fill_value = self.data[column].mode()[0]
            else:
                fill_value = strategy  # 'constant' case with user-defined value

            self.data[column].fillna(fill_value, inplace=True)
        print("Missing values filled.")

    def get_numerical_columns(self):
        """ Returns a list of numerical column names (integer and float) """
        if self.data is not None:
            numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            return numerical_cols
        else:
            print("Data not loaded.")
            return []

    def get_continuous_columns(self):
        """ Returns a list of continuous column names (typically float) """
        if self.data is not None:
            continuous_cols = self.data.select_dtypes(include=['float64']).columns.tolist()
            return continuous_cols
        else:
            print("Data not loaded.")
            return []
        
    def print_numerical_columns(self):
        """ Prints the names of numerical columns in the dataset. """
        numerical_cols = self.get_numerical_columns()
        if numerical_cols:
            print("Numerical columns in the dataset:")
            for col in numerical_cols:
                print(col)
        else:
            print("Data not loaded or no numerical columns found.")

    def print_continuous_columns(self):
        """ Prints the names of continuous columns in the dataset. """
        continuous_cols = self.get_continuous_columns()
        if continuous_cols:
            print("Continuous columns in the dataset:")
            for col in continuous_cols:
                print(col)
        else:
            print("Data not loaded or no continuous columns found.")

    def get_categorical_columns(self):
        """ Returns a list of categorical column names """
        if self.data is not None:
            categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            return categorical_cols
        else:
            print("Data not loaded.")
            return []
        
    def print_categorical_columns(self):
        """ Prints the names of categorical columns in the dataset. """
        categorical_cols = self.get_categorical_columns()
        if categorical_cols:
            print("Categorical columns in the dataset:")
            for col in categorical_cols:
                print(col)
        else:
            print("Data not loaded or no categorical columns found.")

        
    def count_unique_values(self):
        if self.data is not None:
            unique_counts = {col: self.data[col].nunique() for col in self.data.columns}
            return unique_counts
        else:
            print("Data not loaded. Please load data to count unique values.")
            return None
    
    def missing_data_percentage(self):
        if self.data is not None:
            missing_percentage = self.data.isnull().mean() * 100
            return missing_percentage
        else:
            print("Data not loaded. Please load data to calculate missing data percentages.")
            return None
        
    def impute_missing_values(self, strategy='mean', fill_value=None):
        if self.data is not None:
            for column in self.data.columns:
                if self.data[column].dtype == 'object' and (strategy in ['mean', 'median'] or fill_value is not None):
                    fill = fill_value if fill_value is not None else self.data[column].mode()[0]
                elif strategy == 'mean':
                    fill = self.data[column].mean()
                elif strategy == 'median':
                    fill = self.data[column].median()
                else:
                    fill = self.data[column].mode()[0]
                self.data[column].fillna(fill, inplace=True)
            print("Missing values imputed using", strategy)
        else:
            print("Data not loaded. Please load data before imputing missing values.")


    def plot_histogram(self):
        if self.data is not None:
            for column in self.data.columns:
                plt.figure(figsize=(10, 4))
                # Check if the column is numeric or categorical
                if self.data[column].dtype == 'object' or self.data[column].nunique() < 10:
                    # Treat as a categorical column
                    sns.countplot(x=column, data=self.data)
                    plt.title(f'Count Plot of {column}')
                else:
                    # Treat as a numeric column
                    sns.histplot(self.data[column], kde=True)
                    plt.title(f'Histogram of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.xticks(rotation=45)
                plt.tight_layout()  # Adjust layout to make room for rotated labels
                plt.show()
        else:
            print("Data not loaded. Please load data before plotting histograms.")
    def plot_correlation_heatmap(self):
        if self.data is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Correlation Heatmap')
            plt.show()
        else:
            print("Data not loaded.")

    def filter_non_zero(self, columns=None):
        if self.data is not None:
            if columns is None:
                # Apply to all numerical columns by default
                columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            else:
                missing_columns = [col for col in columns if col not in self.data.columns]
                if missing_columns:
                    print(f"Warning: The following specified columns are not in the dataset and will be ignored: {missing_columns}")
                    columns = [col for col in columns if col in self.data.columns]
            # Filter out rows where any of the specified columns have zero value
            self.data = self.data.loc[(self.data[columns] != 0).all(axis=1)]
            print(f"Data filtered. Remaining rows: {len(self.data)}.")
        else:
            print("Data not loaded. Please load data before filtering.")

    def balanced_sample(self, sample_size_per_class):
        if self.target_column and self.target_column in self.data.columns:
            # Group by the target column and sample equal amounts from each group
            grouped = self.data.groupby(self.target_column)
            samples = [grp.sample(n=min(sample_size_per_class, len(grp)), random_state=42) for name, grp in grouped]
            sampled_data = pd.concat(samples)
            self.data = sampled_data
        else:
            print(f"Target column '{self.target_column}' not found or not specified.")

    def remove_outliers(self, target_column=None):
        if self.data is not None:
            if target_column and target_column in self.data.columns:
                data_without_target = self.data.drop(columns=[target_column])
            else:
                data_without_target = self.data

            Q1 = data_without_target.quantile(0.25)
            Q3 = data_without_target.quantile(0.75)
            IQR = Q3 - Q1

            original_size = len(self.data)
            # Filtering out the outliers by keeping only valid values
            self.data = self.data[~((data_without_target < (Q1 - 1.5 * IQR)) | (data_without_target > (Q3 + 1.5 * IQR))).any(axis=1)]
            new_size = len(self.data)

            print(f"Removed {original_size - new_size} outliers. New dataset size: {new_size}")
        else:
            print("Data not loaded. Please load data before removing outliers.")

    def print_identical_rows_count(self):
        if self.data is not None:
            # Identifying all duplicate rows, including the first occurrence
            duplicate_rows = self.data.duplicated(keep=False)
            # Filtering to only include rows that appear more than once
            duplicate_data = self.data[duplicate_rows]

            if not duplicate_data.empty:
                print("Duplicate rows that appear more than once:")
                print(duplicate_data)
            else:
                print("No duplicate rows that appear more than once were found.")
        else:
            print("Data not loaded. Please load data to check for duplicate rows.")


    def print_duplicate_rows_excluding_columns(self, exclude_columns=None):
        if self.data is not None:
            if exclude_columns:
                # Ensure it's a list even if a single column name is provided
                exclude_columns = [exclude_columns] if isinstance(exclude_columns, str) else exclude_columns
                cols_to_use = self.data.columns.difference(exclude_columns)
            else:
                cols_to_use = self.data.columns

            # Finding duplicates based on the filtered columns
            duplicate_mask = self.data.duplicated(subset=cols_to_use, keep=False)
            duplicates = self.data.loc[duplicate_mask]

            if duplicates.empty:
                print("No duplicates found.")
            else:
                grouped = duplicates.groupby(list(cols_to_use))
                group_number = 1
                for name, group in grouped:
                    print(f"{group_number}st duplicate group:")
                    print(group)
                    group_number += 1
        else:
            print("Data not loaded. Please load data before checking for duplicates.")

    def merge_duplicates_assign_majority(self, exclude_columns=None):
        if self.data is not None:
            if exclude_columns:
                exclude_columns = [exclude_columns] if isinstance(exclude_columns, str) else exclude_columns
                data_columns = [col for col in self.data.columns if col not in exclude_columns]

                duplicates = self.data.duplicated(subset=data_columns, keep=False)

                if duplicates.any():
                    duplicate_data = self.data.loc[duplicates]
                    grouped = duplicate_data.groupby(data_columns)

                    for _, group in grouped:
                        if len(group) > 1:
                            for exclude_column in exclude_columns:
                                most_common = group[exclude_column].mode()
                                most_common_value = most_common.iloc[0] if not most_common.empty else group[exclude_column].iloc[0]
                                self.data.loc[group.index, exclude_column] = most_common_value

                    original_count = len(self.data)
                    self.data.drop_duplicates(subset=data_columns, keep='first', inplace=True)
                    new_count = len(self.data)
                    print(f"Original data count: {original_count}, new data count after merging duplicates: {new_count}")
                    print("Duplicates merged and majority value assigned based on the most common value of excluded columns.")
                else:
                    print("No duplicates found to merge.")
            else:
                print("Exclude columns not specified.")
        else:
            print("Data not loaded. Please load data before merging duplicates.")