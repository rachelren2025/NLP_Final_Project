import json
import argparse

def calculate_accuracy(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        correct_predictions = 0
        total_predictions = len(data)
        
        for example in data:
            predicted = int(example["predicted_label"])
            correct = int(example["correct_label"])
            if predicted == correct:
                correct_predictions += 1
        
        accuracy = (correct_predictions / total_predictions)*100 if total_predictions > 0 else 0
        return accuracy
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return None

if __name__ == "__main__":
    accuracy = calculate_accuracy("zlucia_legalbert_casehold_results.json")
    if accuracy is not None:
        print(f"Accuracy: {accuracy:.2f}%")
