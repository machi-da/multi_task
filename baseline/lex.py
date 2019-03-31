from lexrank import LexRank

train_data = '/home/lr/machida/yahoo/work/train.txt.que'
eval_data = '/home/lr/machida/yahoo/work/correct.txt.split'

with open(train_data, 'r')as f:
    data = f.readlines()

print('Load training data')
doc = []
for d in data:
    label, sentences = d.strip().split('\t')
    sentences = sentences.split('|||')
    doc.append(sentences)

print('Training Lexrank')
lxr = LexRank(doc)

correct = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}

with open(eval_data, 'r')as f:
    data = f.readlines()
for d in data:
    label, sentences = d.strip().split('\t')
    sent_num = len(sentences.split('|||')) if len(sentences.split('|||')) <= 6 else 7
    label = int(label)

    sentences = sentences.split('|||')
    score = lxr.rank_sentences(sentences)
    if label == score.argmax() + 1:
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