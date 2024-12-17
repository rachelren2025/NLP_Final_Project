import json
import numpy as np


# each of the three models will output a different file
# e.g. for bert-double, dev_file + quartiles + probabilities + correctness = bert_double_results.json
def combine_parsed_dev_file_with_quartile_probabilities_correctness(quartiles_file, probabilities, parsed_dev_file,
                                                                    predictions,
                                                                    output_file):
    with open(parsed_dev_file, 'r') as f:
        dev_contents = json.load(f)

    with open(quartiles_file, 'r') as g:
        quartiles = json.load(g)

    with open(probabilities, 'r') as h:
        probabilities = h.readlines()

    with open(predictions, 'r') as q:
        predictions = q.readlines()

    probabilities = [x.strip().split(',') for x in probabilities]

    count = 0
    for item in dev_contents:
        item['prediction'] = str(predictions[count])[0]
        item['quartile'] = quartiles[item['id']]
        item['softmax_probabilities'] = [float(y) for y in probabilities[count]]
        if item['prediction'] == item['correct_label']:
            item['newmetric_score'] = calculate_new_metric(True, item['quartile'], item['softmax_probabilities'])
        else:
            item['newmetric_score'] = calculate_new_metric(False, item['quartile'], item['softmax_probabilities'])
        print(item)
        break
        count += 1


# if model is correct, 1 * max(softmax_probabilities) * q1 = 0.25... (rewarded less for easier q)
# if model is wrong, -1 * max(softmax_probabilities) * q1 = 1... (punished more for easier q)
def calculate_new_metric(correct: bool, quartile, probabilities):
    if correct:
        quartile_scores = {
            'q1': 0.25,
            'q2': 0.5,
            'q3': 0.75,
            'q4': 1,
        }

        return max(probabilities) * quartile_scores[quartile]

    else:
        quartile_scores = {
            'q1': 1,
            'q2': 0.75,
            'q3': 0.5,
            'q4': 0.25,
        }

        return -max(probabilities) * quartile_scores[quartile]


bert_double_prob_file = "../Casehold_code/output/bert-double/probabilities.csv"
bert_double_predictions_file = "../Casehold_code/output/bert-double/predictions.csv"


def main():
    combine_parsed_dev_file_with_quartile_probabilities_correctness(
        '../Evaluation_code/split_datasets/GM_sorted_questions.json', bert_double_prob_file, 'parsed_dev_file.json',
        bert_double_predictions_file, 'bert_double_results.json')


main()
