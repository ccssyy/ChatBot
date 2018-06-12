# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 20:54:09 2018

@author: SamChen
"""

word_set={}
def load_word_set(input):
    file_object = open(input, 'r', encoding='utf-8')
    while True:
        line = file_object.readline()
        if line:
            line_pair = line.split('|')
            line_question = line_pair[0].replace("\n",'').strip()
            line_answer = line_pair[1].replace("\n",'').strip()
            for word in line_question.split(' '):
                word_set[word] = 1
            for word in line_answer.split(' '):
                word_set[word] = 1
        else:
            break
    file_object.close()
    
    return word_set

print(load_word_set('./segment_result2.pair'))