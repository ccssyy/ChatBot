# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 21:11:37 2018

@author: SamChen
"""

from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import logging

##设置log日志格式: 时间:级别:消息  级别是INFO以上输出
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
run = 'test'
##拼接字符串 %s代替变量 后面跟上变量
logging.info("running %s begin %s "%(run,'!'))

sentences = word2vec.LineSentence('segment_result.txt')
model = word2vec.Word2Vec(sentences, size=200,workers=8,iter=15,negative=25,window=8,hs=0,sample=1e-4)
model.save('vectors.bin')
#model.wv.save_word2vec_format('vectors.bin',binary=True)
model.wv.save_word2vec_format('vectors.txt',binary=False)

print("Done!!!")

