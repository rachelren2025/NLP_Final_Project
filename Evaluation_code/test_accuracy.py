import pandas as pd
from sklearn.metrics import accuracy_score
import json


def parse_model_results(model1, model2, model3):
    models = [model1, model2, model3]
    predictions = {}
    probabilities = {}
    
    for model in models:
        predictions_path = f"../casehold_code/output/{model}/predictions.csv"
        probabilities_path = f"../casehold_code/output/{model}/probabilities.csv"
        
        # Load predictions and probabilities
        temp_predictions = pd.read_csv(predictions_path, header=None).squeeze().astype(int).tolist()
        temp_probabilities = pd.read_csv(probabilities_path, header=None).values.tolist()
        
        # Store in dictionaries
        predictions[model] = temp_predictions
        probabilities[model] = temp_probabilities
    
    return predictions, probabilities


def calculate_accuracy_by_quarter(model_predictions, answer_key, difficulty_file):
    """
    Calculate accuracy for each difficulty quarter (q1, q2, q3, q4) and overall.
    Args:
        model_predictions: List of model predictions.
        answer_key: List of true labels.
        difficulty_file: JSON file containing question difficulty classification.
    Returns:
        Dictionary with accuracies for each quarter and overall.
    """

    # Load difficulty classification
    with open(difficulty_file, "r") as f:
        difficulty_data = json.load(f)

    # Extract relevant indices and limit based on difficulty file length
    max_index = len(difficulty_data)
    model_predictions = model_predictions[:max_index]
    answer_key = answer_key[:max_index]

    # Separate indices for each quarter
    quarter_indices = {
        "q1": [i for i, k in enumerate(difficulty_data.keys()) if difficulty_data[k] == "q1"],
        "q2": [i for i, k in enumerate(difficulty_data.keys()) if difficulty_data[k] == "q2"],
        "q3": [i for i, k in enumerate(difficulty_data.keys()) if difficulty_data[k] == "q3"],
        "q4": [i for i, k in enumerate(difficulty_data.keys()) if difficulty_data[k] == "q4"],
    }

    # Calculate accuracies for each quarter
    accuracies = {}
    for quarter, indices in quarter_indices.items():
        quarter_predictions = [model_predictions[i] for i in indices]
        quarter_answers = [answer_key[i] for i in indices]
        quarter_accuracy = accuracy_score(quarter_answers, quarter_predictions) if quarter_predictions else 0.0
        accuracies[f"{quarter}_accuracy"] = quarter_accuracy

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(answer_key, model_predictions)
    accuracies["overall_accuracy"] = overall_accuracy

    return accuracies


if __name__ == "__main__":
    # Load Answer Key
    answer_key = pd.read_csv("../casehold_code/data/dev.csv", usecols=[12]).squeeze().astype(int).tolist()

    # Load Model Results
    model_predictions, model_probabilities = parse_model_results("bert-double", "legal-bert", "custom-legal-bert")

    # Difficulty files from the previous prompt
    difficulty_files = [
        "split_datasets/GM_sorted_questions.json",  # Geometric Mean
    ]

    for difficulty_file in difficulty_files:
        sorting_method = "Geometric Mean"

        for model, predictions in model_predictions.items():
            print(f"Model: {model}, Sorted By: {sorting_method}")

            accuracies = calculate_accuracy_by_quarter(predictions, answer_key, difficulty_file)
            print(f"   Q1 Questions Accuracy: {accuracies['q1_accuracy'] * 100:.2f}%")
            print(f"   Q2 Questions Accuracy: {accuracies['q2_accuracy'] * 100:.2f}%")
            print(f"   Q3 Questions Accuracy: {accuracies['q3_accuracy'] * 100:.2f}%")
            print(f"   Q4 Questions Accuracy: {accuracies['q4_accuracy'] * 100:.2f}%")
            print(f"   Overall Accuracy: {accuracies['overall_accuracy'] * 100:.2f}%")
