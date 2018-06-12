# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:30:29 2018

@author: SamChen
"""

import math
import numpy as np
import os
import pickle

def load_vector(input,output):
    print('begin loading vectors ......')

    word_vector = {}
    if os.path.exists(output):
        print(output+" exist!")
        file = open(output,"rb")
        word_vector = pickle.load(file)
    else:
        print(output+" not exist!")
        input_file = open(input, "r", encoding='utf-8')
        output_file = open(output,"wb")
    
        #获取词表的数目及向量维度
        words_and_size = input_file.readline()
        words_and_size = words_and_size.strip()
        words = int(words_and_size.split(' ')[0])
        size = int(words_and_size.split(' ')[1])
        print('词数：%d' % words)
        print('维度：%d' % size)

        weight_str = ''
    
        for b in range(0, words):
            print('正在处理第 %d/%d 个词......%.2f%%' % (b+1,words,float(((b+1)/words)*100)))
        
            line = input_file.readline()
            line = line.strip()
            word = line.split(' ', 1)[0]
            line2 = line.split(' ',1)[1]
            vector = np.empty([200])
            #print(word)
            #print(line2)
            #print(line2.split(' '))
            for index in range(0,size):
                weight_str = line2.split(' ')[index]
                weight = float(weight_str)
                #print(weight)
                vector[index] = weight
            #print(vector)
    
            #将词与对应向量存到dict中
            word_vector[word] = vector
        
        input_file.close()
    
        #print(word_vector)
        print('loading vectors finished !!!')
        #output_file.write(str(word_vector))
        pickle.dump(word_vector,output_file)
        output_file.close()
    
    return word_vector

d = load_vector('vectors.txt','word_vector_model2.txt')
#print(d[u'我'])
#print(d[u'大坂西'])
print(d['大坂西'])
