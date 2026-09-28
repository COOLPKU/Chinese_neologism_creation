import json

exiting_word_set = set()
with open('rule_based_process/union_words_disyllabic.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        exiting_word_set.add(line.strip())

with open('rule_based_process/filtered_prediction_pairs.txt', 'r', encoding='utf-8') as f:
    all_words_to_process = json.load(f)


