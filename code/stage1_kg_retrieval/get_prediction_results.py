import json

with open('discriminate/data/predictions.json','r',encoding='utf-8') as f:
    predictions = json.load(f)

with open('discriminate/data/filtered_pairs.txt','r',encoding='utf-8') as f:
    filtered_pairs = f.readlines()

morph_dict = {}
with open('discriminate/data/1_语素.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line = line.split()
        morph_dict[line[0]]=line

result_list = []

print(len(filtered_pairs),len(predictions))

for pair, prediction in zip(filtered_pairs, predictions):
    head_id, relation, tail_id = pair.strip().split('\t')
    head_pos = morph_dict[head_id[1:]][4].replace('素@待定','')
    head_sense = morph_dict[head_id[1:]][6]
    tail_pos = morph_dict[tail_id[1:]][4].replace('素@待定','')
    tail_sense = morph_dict[tail_id[1:]][6]
    result = prediction['prediction']
    result_list.append({
        'head_id': head_id,
        'head_pos': head_pos,
        'head_sense': head_sense,

        'relation': relation,
        'tail_id': tail_id,
        'tail_pos': tail_pos,
        'tail_sense': tail_sense,
        'prediction': result,
        'confidence': prediction['confidence']
    })
print(len(result_list))
with open('discriminate/data/filtered_prediction_pairs_all.json', 'w',encoding='utf-8') as f:
    json.dump(result_list, f, indent=4,ensure_ascii=False)