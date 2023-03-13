
# -*- coding: gb2312 -*-

import  jieba, sys, numpy, os
import jieba.analyse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import word2vec
from gensim.models import KeyedVectors

filepath = "./corpus/�ƽ�ʱ��.txt"

with open(filepath,encoding='utf-8') as f:
    document = f.read()
    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)
    print("type",type(result))
    with open('./�ƽ�ʱ��_segment.txt', 'w',encoding="utf-8") as f2:
        f2.write(result)

sentences = word2vec.LineSentence('./�ƽ�ʱ��_segment.txt')
path = get_tmpfile("w2v_model.bin") #������ʱ�ļ�
model = word2vec(sentences, size=200, window=5, min_count=1)
#ģ�ʹ��������1
model.save(path)
model=Word2Vec.load("w2v_model.bin")
for key in model.similar_by_word('����',topn=10):
        print(key)
#ģ�ʹ��������2
path1 = get_tmpfile("w2v_vector.bin") #������ʱ�ļ�
model.wv.save(path1)
wv = KeyedVectors.load("w2v_vector.bin", mmap='r')
for key in wv.similar_by_word('��', topn =10):
    print(key)