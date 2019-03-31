import math
import copy


train_que_data = '/home/lr/machida/yahoo/work/train.txt.que'
train_ans_data = '/home/lr/machida/yahoo/work/train.txt.ans'
eval_data = '/home/lr/machida/yahoo/work/correct.txt.split'

with open(train_que_data, 'r')as f:
    data = f.readlines()
data_size = len(data)
# IDF辞書
df_dic = {}
for d in data:
    sentences = d.split('\t')[1]
    sentences = sentences.replace('|||', ' ')
    words = sentences.split(' ')
    for word in words:
        if word not in df_dic:
            df_dic[word] = 1
        else:
            df_dic[word] += 1

correct = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}

with open(eval_data, 'r')as f:
    data = f.readlines()
for d in data:
    label, sentences = d.strip().split('\t')
    label = int(label)

    score_dic = {}
    sent_num = len(sentences.split('|||')) if len(sentences.split('|||')) <= 6 else 7
    for i, sentence in enumerate(sentences.split('|||'), start=1):
        idf_score = 0
        for word in sentence.split(' '):
            if word in df_dic:
                # print(word, data_size, df_dic[word])
                idf_score += math.log(data_size / df_dic[word])
        score_dic[i] = idf_score / len(sentence.split(' '))
    predict_label = max(score_dic, key=lambda x: score_dic[x])

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

print('que')  # acl shortではこちらを使用した
print('micro: {}, macro: {}, {}'.format(round(micro[0]/micro[1], 3), ', '.join(macro), round(sum(macro_ave)/len(macro_ave), 3)))

df_qa_dic = copy.deepcopy(df_dic)
qa_data_size = data_size


with open(train_ans_data, 'r')as f:
    data = f.readlines()
data_size = len(data)
qa_data_size += data_size
# IDF辞書
df_dic = {}
for d in data:
    words = d.strip().split(' ')
    for word in words:
        if word not in df_dic:
            df_dic[word] = 1
        else:
            df_dic[word] += 1

        if word not in df_qa_dic:
            df_qa_dic[word] = 1
        else:
            df_qa_dic[word] += 1

correct = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}

with open(eval_data, 'r')as f:
    data = f.readlines()
for d in data:
    label, sentences = d.strip().split('\t')
    label = int(label)

    score_dic = {}
    sent_num = len(sentences.split('|||')) if len(sentences.split('|||')) <= 6 else 7
    for i, sentence in enumerate(sentences.split('|||'), start=1):
        idf_score = 0
        for word in sentence.split(' '):
            if word in df_dic:
                idf_score += math.log(data_size / df_dic[word])
        score_dic[i] = idf_score / len(sentence.split(' '))
    predict_label = max(score_dic, key=lambda x: score_dic[x])

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

print('ans')
print('micro: {}, macro: {}, {}'.format(round(micro[0]/micro[1], 3), ', '.join(macro), round(sum(macro_ave)/len(macro_ave), 3)))


with open(eval_data, 'r')as f:
    data = f.readlines()
for d in data:
    label, sentences = d.strip().split('\t')
    label = int(label)

    score_dic = {}
    sent_num = len(sentences.split('|||')) if len(sentences.split('|||')) <= 6 else 7
    for i, sentence in enumerate(sentences.split('|||'), start=1):
        idf_score = 0
        for word in sentence.split(' '):
            if word in df_qa_dic:
                idf_score += math.log(qa_data_size / df_qa_dic[word])
        score_dic[i] = idf_score / len(sentence.split(' '))
    predict_label = max(score_dic, key=lambda x: score_dic[x])

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

print('que+ans')
print('micro: {}, macro: {}, {}'.format(round(micro[0]/micro[1], 3), ', '.join(macro), round(sum(macro_ave)/len(macro_ave), 3)))