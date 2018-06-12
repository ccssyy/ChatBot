# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:41:07 2018

@author: SamChen
"""

from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
translate='translate'
logging.info("running %s begin %s"%(translate,'!'))

new_model = word2vec.Word2Vec.load('vectors.model')
new_model.wv.save_word2vec_format('vectors.bin',binary=True)
new_model.wv.save_word2vec_format('vectors.txt',binary=False)
print('Translate Done!!!')