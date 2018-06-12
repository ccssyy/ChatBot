# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:43:54 2018

@author: SamChen
"""

import gensim
from gensim.models import word2vec

new_model = word2vec.Word2Vec.load('vectors.bin')
#binary_model = gensim.models.KeyedVectors.load_word2vec_format('vectors.bin')
##txt_model = gensim.models.KeyedVectors.load_word2vec_format('vectors.txt')

print(new_model.wv.most_similar("滋润"))
#print(binary_model.wv.most_similar("滋润"))
#print(txt_model.wv.most_similar("滋润"))
print(new_model.wv.most_similar("兔子"))
#print(binary_model.wv.most_similar("兔子"))
#print(txt_model.wv.most_similar("兔子"))
print(new_model.wv.most_similar("爸爸"))
print(new_model.wv.most_similar("你好"))
print(new_model.wv.most_similar("电脑"))
print(new_model.wv.most_similar("物理"))

