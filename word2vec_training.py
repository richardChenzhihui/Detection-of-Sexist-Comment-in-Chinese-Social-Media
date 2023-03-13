from gensim.models import word2vec
import numpy as np

sentences = word2vec.LineSentence('./corupus_seg.txt')
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, vector_size=100, negative = 1)

model.save('word2vec.model')

model.wv.save_word2vec_format('word2vec.txt')

# 第一种
# model = Word2Vec.load(word2vec.model)
# model.save('word2vec.model')
# # 第二种
# model = gensim.models.KeyedVectors.load_word2vec_format('word2vec.bin',binary=True)
# model.wv.save_word2vec_format('word2vec.bin')
# # 第三种
# gensim.models.KeyedVectors.load_word2vec_format('word2vec.txt',binary=False)
# model.wv.save_word2vec_format('word2vec.txt')

