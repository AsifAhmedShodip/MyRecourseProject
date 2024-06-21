
import sys  
sys.path.insert(1, '/Users/asifahmed/Documents/Codes/MyRecourseProject')

import pandas as pd
import itertools
from models.model_trainer import ModelTrainer
from data_handling.dataset import Dataset
from evaluation.evaluator import Evaluator

def is_numerical(data, column, target_column, threshold=20):
    """
    Consider a column numerical if it's not the target and has more than `threshold` unique values.
    """
    return data[column].nunique() > threshold and column != target_column


def automated_evaluation(file_path, target_column, model_type='svm', threshold=0.6, sample_size=300):
    # Initialize the model trainer
    trainer = ModelTrainer()

    dataset_name = file_path.split("/")[-1]
     # Initialize results list
    passed_pairs = []

    # Get all truly numerical columns from the original data
    original_data_instance = Dataset(target_column=target_column)
    original_data_instance.load_csv(file_path=file_path)
    original_data_instance.encode_categorical_columns()
    # original_data_instance.remove_outliers()
    original_data_instance.balanced_sample(sample_size)

    numerical_columns = [col for col in original_data_instance.data.columns if is_numerical(original_data_instance.data, col, target_column)]

    # Iterate over all pairs of numerical features
    for feature1, feature2 in itertools.combinations(numerical_columns, 2):
        # Reload data instance for each pair
        data_instance = Dataset(target_column=target_column)
        data_instance.load_csv(file_path)
        data_instance.encode_categorical_columns()
        # data_instance.remove_outliers()
        data_instance.balanced_sample(sample_size)

        # Select features and ensure they are present in the dataset
        data_instance.select_features([feature1, feature2, target_column])
        
        # if data_instance.data.empty:
        #     continue
        
        # Split and scale the data
        X_train, X_test, y_train, y_test = trainer.split_data(data_instance.data, target_column=target_column)

        if X_train.empty or X_test.empty:
            continue

        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
        
        # Train the model
        model = trainer.train(model_type, X_train_scaled, y_train)
        
        # Create an evaluator with the trained model and test data
        evaluator = Evaluator(model, X_test_scaled, y_test)
        
        # Obtain metrics using the Evaluator methods
        metrics = evaluator.get_evaluation_metrics()

        # Print necessary information if the threshold is met
        if metrics['Accuracy'] and metrics['Accuracy'] >= threshold:
            print(f"\033[1;32mPassed: {feature1}, {feature2} with accuracy {metrics['Accuracy']:.2f}\033[0m")
            evaluator.report()
            passed_pairs.append({
                'Dataset': dataset_name,
                'Feature1': feature1, 
                'Feature2': feature2,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'], 
                'Recall': metrics['Recall'], 
                'F1 Score': metrics['F1 Score'],
                'Confusion Matrix': metrics['Confusion Matrix'].tolist(),
                'Classification Report': metrics['Classification Report']
            })
        else:
            # Print necessary information for all pairs
            print(f"Failed: {feature1}, {feature2} with accuracy {metrics['Accuracy']:.2f}")

        # Add space between iterations for readability
        print("\n" + "-" * 50 + "\n")

# Save passed pairs information to a text file]
    print("Saving passed pairs information to 'evaluation_results.txt'...")
    with open('evaluation_results.txt', 'w') as file:
        for pair in passed_pairs:
            file.write(f"Dataset: {pair['Dataset']}\n")
            file.write(f"Features: {pair['Feature1']}, {pair['Feature2']}\n")
            file.write(f"Accuracy: {pair['Accuracy']:.2f}\n")
            file.write(f"Precision: {pair['Precision']:.2f}\n")
            file.write(f"Recall: {pair['Recall']:.2f}\n")
            file.write(f"F1 Score: {pair['F1 Score']:.2f}\n")
            file.write(f"Confusion Matrix: {pair['Confusion Matrix']}\n")
            file.write("Classification Report:\n")
            file.write(f"{pair['Classification Report']}\n")
            file.write("-" * 50 + "\n")

automated_evaluation('/Users/asifahmed/Documents/Codes/MyRecourseProject/datasets/processed/credit_processed.csv', 
                     target_column='NoDefaultNextMonth',
                     threshold=.65,
                     sample_size=1000)