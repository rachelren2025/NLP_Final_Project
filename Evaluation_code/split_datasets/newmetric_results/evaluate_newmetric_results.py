import json

with open('custom_legal_bert_results.json', 'r') as f:
    k = json.load(f)

sum_nm = 0
l = len(k)

for j in k:
    sum_nm += j['newmetric_score']

print(sum_nm/l)

# double -0.44
# legal -0.48
# custom-legal -0.36

