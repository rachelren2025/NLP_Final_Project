import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

def parse_model_results(model1, model2, model3):
    models = [model1, model2, model3]
    predictions = {}
    probabilities = {}

    for model in models:
        predictions_path = f"output/{model}/predictions.csv"
        probabilities_path = f"output/{model}/probabilities.csv"

        # Load predictions and probabilities
        temp_predictions = pd.read_csv(predictions_path, header=None).squeeze().astype(int).tolist()
        temp_probabilities = pd.read_csv(probabilities_path, header=None).values.tolist()

        # Store in dictionaries
        predictions[model] = temp_predictions
        probabilities[model] = temp_probabilities

    return predictions, probabilities

def compute_accuracy(predictions, answer_key):
    if len(predictions) != len(answer_key):
        raise ValueError("The length of predictions and answer_key must match.")

    accuracy = accuracy_score(answer_key, predictions)
    return accuracy

def compute_confidence_metrics(predictions, probabilities, answer_key, threshold=0.7):
    confident_predictions = []
    confident_answers = []
    unconfident_predictions = []
    unconfident_answers = []

    for pred, prob, true_ans in zip(predictions, probabilities, answer_key):
        if max(prob) >= threshold:  # Confident predictions
            confident_predictions.append(pred)
            confident_answers.append(true_ans)
        else:  # Unconfident predictions
            unconfident_predictions.append(pred)
            unconfident_answers.append(true_ans)

    # Compute metrics for confident predictions
    confident_accuracy = accuracy_score(confident_answers, confident_predictions) if confident_predictions else 0.0
    confident_percentage = len(confident_answers) / len(predictions) if predictions else 0.0

    # Compute metrics for unconfident predictions
    unconfident_accuracy = accuracy_score(unconfident_answers, unconfident_predictions) if unconfident_predictions else 0.0
    unconfident_percentage = len(unconfident_answers) / len(predictions) if predictions else 0.0

    return confident_accuracy, confident_percentage, unconfident_accuracy, unconfident_percentage

if __name__ == "__main__":
    # Load Answer Key
    answer_key = pd.read_csv("data/dev.csv", usecols=[12]).squeeze().astype(int).tolist()

    # Load Model Results
    model_predictions, model_probabilities = parse_model_results("bert-double", "legal-bert", "custom-legal-bert")

    for model, predictions in model_predictions.items():
        print(f"Model: {model}")

        # Standard Accuracy
        accuracy = compute_accuracy(predictions, answer_key)
        print(f"   Standard Accuracy: {accuracy * 100:.2f}%")

        # Confidence Metrics
        confidence_threshold = 0.75  # Set desired probability threshold here
        confident_accuracy, confident_percentage, unconfident_accuracy, unconfident_percentage = compute_confidence_metrics(
            model_predictions[model], model_probabilities[model], answer_key, confidence_threshold
        )

        print(f"   Confident Accuracy (probability >= {confidence_threshold}): {confident_accuracy * 100:.2f}%")
        print(f"        Percentage of confident questions: {confident_percentage * 100:.2f}%")

        print(f"   Unconfident Accuracy (probability < {confidence_threshold}): {unconfident_accuracy * 100:.2f}%")
        print(f"        Percentage of unconfident questions: {unconfident_percentage * 100:.2f}%")
