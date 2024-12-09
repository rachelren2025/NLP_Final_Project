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

# Computes accuracy only for questions where the highest probability is greater than the threshold
def compute_confidence_threshold(predictions, probabilities, answer_key, threshold=0.7):
    filtered_predictions = []
    filtered_answers = []
    
    for pred, prob, true_ans in zip(predictions, probabilities, answer_key):
        if max(prob) >= threshold:  # Check if the highest probability exceeds the threshold
            filtered_predictions.append(pred)
            filtered_answers.append(true_ans)
    
    if not filtered_predictions:
        return 0.0  # No valid predictions
    
    percentage = len(filtered_answers)/len(predictions)

    return percentage, accuracy_score(filtered_answers, filtered_predictions)
# Not using log loss
def compute_question_log_loss(probabilities, true_label):
    """
    Calculate Log Loss for a single question.
    
    Parameters:
    - probabilities: List of predicted probabilities for each class.
    - true_label: Integer index of the correct answer.
    
    Returns:
    - log_loss_value: Log loss for the question.
    """
    num_classes = len(probabilities)
    
    # Normalize probabilities to ensure they sum to 1
    probabilities = [p / sum(probabilities) for p in probabilities]
    
    # Convert true label to one-hot encoding
    one_hot = [0] * num_classes
    one_hot[true_label] = 1

    # Compute log loss
    return log_loss([one_hot], [probabilities])
# Not using log loss
def compute_confidence_based_on_log_loss(probabilities, true_labels, threshold):
    """
    Filter questions based on log loss confidence.
    
    Parameters:
    - probabilities: List of lists of probabilities for each question.
    - true_labels: List of true labels (indices of correct answers).
    - threshold: Log loss threshold to filter confident predictions.
    
    Returns:
    - percentage: Percentage of confident questions.
    - accuracy: Accuracy for confident questions.
    """
    confident_predictions = []
    confident_answers = []
    model_log_loss_values =[]
    
    for probs, true_label in zip(probabilities, true_labels):
        log_loss_value = compute_question_log_loss(probs, true_label)
        model_log_loss_values.append(log_loss_value)
        if log_loss_value <= threshold:  # Lower log loss indicates higher confidence
            confident_predictions.append(probs.index(max(probs)))  # Predicted label
            confident_answers.append(true_label)
    
    if not confident_predictions:
        return 0.0, 0.0  # No confident predictions
    
    percentage = len(confident_answers) / len(probabilities)
    accuracy = accuracy_score(confident_answers, confident_predictions)
    #return model_log_loss_values to have logloss score for every question
    return model_log_loss_values, percentage, accuracy

if __name__ == "__main__":
    # Load Answer Key
    answer_key = pd.read_csv("data/dev.csv", usecols=[12]).squeeze().astype(int).tolist()

    # Load Model Results
    model_predictions, model_probabilities = parse_model_results("bert-double", "legal-bert", "custom-legal-bert")
    '''
    model_predictions = {
        "bert-double": [1, 0, 1, ... 2, 0],
        "legal-bert": [0, 1, 1, ... 2, 2],
        "custom-legal-bert": [2, 0, 0, ... 1, 1]
    }
    model_probabilities = {
        "bert-double": [
            [0.1, 0.7, 0.3, 0.2 0.2],  # Probabilities for Question 1
            [0.8, 0.1, 0.1],  # Probabilities for Question 2
            [0.3, 0.5, 0.2],  # Probabilities for Question 3
            [0.2, 0.3, 0.5],  # Probabilities for Question 4
            [0.6, 0.3, 0.1]   # Probabilities for Question 5
        ],
        "legal-bert": [
            [0.4, 0.4, 0.2],
            [0.1, 0.8, 0.1],
            [0.3, 0.6, 0.1],
            [0.2, 0.5, 0.3],
            [0.5, 0.3, 0.2]
        ],
        "custom-legal-bert": [
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.6, 0.2, 0.2],
            [0.4, 0.4, 0.2],
            [0.1, 0.7, 0.2]
        ]
    }
    '''
    
    for model, predictions in model_predictions.items():
        print(f"Model: {model}")
        
        accuracy = compute_accuracy(predictions, answer_key)
        print(f"   Standard Accuracy: {accuracy * 100:.2f}%")
        
        confidence_threshold = 0.75 # Set desired probability threshold here
        percentage, filtered_accuracy = compute_confidence_threshold(model_predictions[model], model_probabilities[model], answer_key, confidence_threshold)
        print(f"   Confident Accuracy (probability <= {confidence_threshold}): {filtered_accuracy * 100:.2f}%")
        print(f"        Percentage of confident questions: {percentage * 100:.2f}%")

        # Logloss Computation - better confidence score option
        # log_loss_threshold = 1.5
        # actuallogloss, percentage, confident_accuracy = compute_confidence_based_on_log_loss(model_probabilities[model], answer_key, log_loss_threshold)
        # print(f"   Confident Accuracy (Log Loss <= {log_loss_threshold}): {confident_accuracy * 100:.2f}%")
        # print(f"       Percentage of confident questions: {percentage * 100:.2f}%")
        # if model == "bert-double":
        #     print(actuallogloss)
