import json
pairs = []
with open('new_word\\data\\predictions.json', 'r', encoding='utf-8') as f:
    predictions = json.load(f)

morph_dict={}
with open('new_word\\data\\morpheme_appear_time.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.split()
        morph_dict[line[0]] = line

morph_appear_dict = {}
with open('new_word\\data\\morpheme_appear_time.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.split()
        morph_appear_dict[line[0]] = int(line[1])

for item in predictions:
    head_id, relation = item['head'], item['relation']
    for predict in item['predictions']:
        predict_id = predict['entity']
        if predict_id[1:] in morph_appear_dict and morph_appear_dict[predict_id[1:]]>=10:
            pairs.append((head_id, relation, predict_id))

with open('new_word\\data\\filtered_pairs.txt', 'w', encoding='utf-8') as f:
    for pair in pairs:
        f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\n")