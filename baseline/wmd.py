from gensim.models import KeyedVectors

train_data = '/home/lr/machida/yahoo/work/train.txt.que'
eval_data = '/home/lr/machida/yahoo/work/correct.txt.split'

print('Load w2v model')
model = '/home/lr/machida/yahoo/distbase/que_best.txt.w2v.bin'
embeddings_model = KeyedVectors.load_word2vec_format(model, binary=True)


correct = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}

with open(eval_data, 'r')as f:
    data = f.readlines()

for d in data:
    label, sentences = d.strip().split('\t')
    label = int(label)
    sent_num = len(sentences.split('|||')) if len(sentences.split('|||')) <= 6 else 7

    entire_sentence = sentences.replace('|||', ' ')

    score_dic = {}
    for i, sentence in enumerate(sentences.split('|||'), start=1):
        distance = embeddings_model.wmdistance(entire_sentence, sentence)
        score_dic[i] = distance
    predict_label = min(score_dic, key=lambda x: score_dic[x])

    if label == predict_label:
        correct[sent_num][0] += 1
    correct[sent_num][1] += 1

micro = [0, 0]
macro = []
macro_ave = []
for k, v in correct.items():
    micro[0] += v[0]
    micro[1] += v[1]
    macro.append('{}: {}'.format(k, round(v[0]/v[1], 3)))
    macro_ave.append(v[0]/v[1])

print('micro: {}, macro: {}, {}'.format(round(micro[0]/micro[1], 3), ', '.join(macro), round(sum(macro_ave)/len(macro_ave), 3)))