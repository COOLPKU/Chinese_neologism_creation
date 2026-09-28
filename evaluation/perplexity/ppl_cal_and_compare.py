import json
import statistics
with open('discriminate/rule_based_process/filtered_prediction_pairs.json', 'r', encoding='utf-8') as f:
    filter_data = json.load(f)
    filtered_word_pairs = set()
    for item in filter_data:
        char1 = item['head_id']
        char2 = item['tail_id']
        wf = item['relation']
        filtered_word_pairs.add((char1,char2,wf))

model_list = ['_bge_large_zh_v1.5','_Qwen3-1.7B','_Llama-3.2-3B-Instruct','']
for model in model_list:
    print(f"Model: {model if model else 'modernbert'}")
    with open(f'discriminate/perplexity_calculate/positive_words_with_sentences_all_masked_ppl{model}.json', 'r', encoding='utf-8') as f:
        positive_word_pool = json.load(f)
        positive_word_ppl = [item['mean_perplexity'] for item in positive_word_pool if item['mean_perplexity']!=float('inf')]

    with open(f'discriminate/perplexity_calculate/negative_words_with_sentences_all_masked_ppl{model}.json', 'r', encoding='utf-8') as f:
        negative_word_pool = json.load(f)
        negative_word_ppl = [item['mean_perplexity'] for item in negative_word_pool if item['mean_perplexity']!=float('inf')]

        negative_word_ppl_advanced = [item['mean_perplexity'] for item in negative_word_pool if (item['head_id'],item['tail_id'],item['relation']) not in filtered_word_pairs and item['mean_perplexity']!=float('inf')]


    with open(f'discriminate/perplexity_calculate/existing_words_with_sentences_all_masked_ppl{model}.json', 'r', encoding='utf-8') as f:
        existing_word_pool = json.load(f)
        existing_word_ppl = [item['mean_perplexity'] for item in existing_word_pool if item['mean_perplexity']!=float('inf')]

    # existing_word=set()
    # with open('discriminate/rule_based_process/union_words_disyllabic.txt', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         existing_word.add(line.strip())

    # with open('discriminate/rule_based_process/filtered_prediction_pairs.json', 'r', encoding='utf-8') as f:
    #     total_word_pool = json.load(f)

    # with open('discriminate/rule_based_process/filtered_prediction_pairs_step4.json', 'r', encoding='utf-8'  ) as f:
    #     positive_word_pool = json.load(f)

    # with open('discriminate/perplexity_calculate/ppl_part1.json', 'r', encoding='utf-8') as f:
    #     ppl_word_pool = json.load(f)

    # positive_words = set()
    # negative_words = set()

    # for item in positive_word_pool:
    #     char1 = item['head_id'][0]
    #     char2 = item['tail_id'][0]
    #     wf = item['relation']
    #     positive_words.add((char1,char2,wf))

    # for item in total_word_pool:
    #     char1 = item['head_id'][0]
    #     char2 = item['tail_id'][0]
    #     wf = item['relation']
    #     if char1+char2 in existing_word:
    #         continue
    #     if (char1,char2,wf) in positive_words:
    #         continue
    #     negative_words.add((char1,char2,wf))



    # positive_ppl=[]
    # negative_ppl=[]
    # for item in ppl_word_pool:
    #     char1 = item['head_id'][0]
    #     char2 = item['tail_id'][0]
    #     wf = item['relation']
    #     if (char1,char2,wf) in positive_words:
    #         positive_ppl.append(item['mean_perplexity'])
    #     elif (char1,char2,wf) in negative_words:
    #         negative_ppl.append(item['mean_perplexity'])


    # Calculate statistics for positive_ppl
    if positive_word_ppl:
        pos_mean = statistics.geometric_mean(positive_word_ppl)
        pos_variance = statistics.variance(positive_word_ppl)
        print(f"Positive PPL - Mean: {pos_mean:.4f}, Variance: {pos_variance:.4f}")
    else:
        print("Positive PPL - No data")

    # Calculate statistics for negative_ppl
    if negative_word_ppl:
        neg_mean = statistics.geometric_mean(negative_word_ppl)
        neg_variance = statistics.variance(negative_word_ppl)
        print(f"Negative PPL - Mean: {neg_mean:.4f}, Variance: {neg_variance:.4f}")
    else:
        print("Negative PPL - No data")

    if negative_word_ppl_advanced:
        neg_mean = statistics.geometric_mean(negative_word_ppl_advanced)
        neg_variance = statistics.variance(negative_word_ppl_advanced)
        print(f"Negative PPL (Advanced) - Mean: {neg_mean:.4f}, Variance: {neg_variance:.4f}")
    else:
        print("Negative PPL (Advanced) - No data")

    if existing_word_ppl:
        exist_mean = statistics.geometric_mean(existing_word_ppl)
        exist_variance = statistics.variance(existing_word_ppl)
        print(f"Existing Word PPL - Mean: {exist_mean:.4f}, Variance: {exist_variance:.4f}")