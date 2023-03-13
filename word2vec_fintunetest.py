from gensim.models import word2vec
import numpy as np


model = word2vec.Word2Vec.load('word2vec.model')


# import sys
# more_sentences=[]
# with open('./SexHateLex.txt','r') as f:
# 	for line in f:
# 		more_sentences.append(list(line.strip('\n').split(',')))
#
# model.build_vocab(more_sentences, update=True)
# model.train(more_sentences, total_examples=model.corpus_count, epochs=model.epochs)


role1 = ['女性','平权','羞耻']
role2 = ['男人','肮脏']
pairs = [(x,y) for x in role1 for y in role2]
# “大鸡巴”只出现于SexHateLex 这个新语料库中，fintune使得word2vec包含该词汇
#print(model.wv['大鸡巴'])

print(model.wv.most_similar('拳师',topn = 20))
print(model.wv.most_similar('平权',topn = 20))
print(np.dot(model.wv['平权'],model.wv['男权']))
print(np.dot(model.wv['平权'],model.wv['女权']))
print(np.dot(model.wv['女权'],model.wv['政治']))
print(np.dot(model.wv['房子'],model.wv['政治']))
print(np.dot(model.wv['女权'],model.wv['拳师']))

print(np.dot(model.wv['女权'],model.wv['父权']))
print(np.dot(model.wv['房子'],model.wv['鼠标']))
print(np.linalg.norm(model.wv['房子']))