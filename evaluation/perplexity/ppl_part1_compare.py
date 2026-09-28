import json
import statistics

existing_word=set()
with open('discriminate/rule_based_process/union_words_disyllabic.txt', 'r', encoding='utf-8') as f:
    for line in f:
        existing_word.add(line.strip())

with open('discriminate/rule_based_process/filtered_prediction_pairs.json', 'r', encoding='utf-8') as f:
    total_word_pool = json.load(f)

with open('discriminate/rule_based_process/filtered_prediction_pairs_step4.json', 'r', encoding='utf-8'  ) as f:
    positive_word_pool = json.load(f)

with open('discriminate/perplexity_calculate/ppl_part1.json', 'r', encoding='utf-8') as f:
    ppl_word_pool = json.load(f)

positive_words = set()
negative_words = set()

for item in positive_word_pool:
    char1 = item['head_id'][0]
    char2 = item['tail_id'][0]
    wf = item['relation']
    positive_words.add((char1,char2,wf))

for item in total_word_pool:
    char1 = item['head_id'][0]
    char2 = item['tail_id'][0]
    wf = item['relation']
    if char1+char2 in existing_word:
        continue
    if (char1,char2,wf) in positive_words:
        continue
    negative_words.add((char1,char2,wf))



positive_ppl=[]
negative_ppl=[]
for item in ppl_word_pool:
    char1 = item['head_id'][0]
    char2 = item['tail_id'][0]
    wf = item['relation']
    if (char1,char2,wf) in positive_words:
        positive_ppl.append(item['mean_perplexity'])
    elif (char1,char2,wf) in negative_words:
        negative_ppl.append(item['mean_perplexity'])


# Calculate statistics for positive_ppl
if positive_ppl:
    pos_mean = statistics.mean(positive_ppl)
    pos_variance = statistics.variance(positive_ppl)
    print(f"Positive PPL - Mean: {pos_mean:.4f}, Variance: {pos_variance:.4f}")
else:
    print("Positive PPL - No data")

# Calculate statistics for negative_ppl
if negative_ppl:
    neg_mean = statistics.mean(negative_ppl)
    neg_variance = statistics.variance(negative_ppl)
    print(f"Negative PPL - Mean: {neg_mean:.4f}, Variance: {neg_variance:.4f}")
else:
    print("Negative PPL - No data")