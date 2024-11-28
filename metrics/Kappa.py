from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(list(answer_key.values()), list(system_output.values()))
