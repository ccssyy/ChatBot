# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 12:03:43 2018

@author: SamChen
"""
import jieba
from jieba import analyse

def segment2pair(input, output):
    input_file = open(input, "r", encoding='utf-8')
    output_file = open(output, "a", encoding='utf-8')
    #count = len(input_file.readlines())
    i = 0
    while True:
        line = input_file.readline()
        if line:
            line = line.strip()
            seg_list = jieba.cut(line)
            segments = ""
            for str in seg_list:
                segments = segments + " " + str
            if i%2 == 0:
                question = segments + " |"
                i=i+1
                continue
            else:
                answer = segments + '\n'
                i=i+1
                output_file.write(question+answer)
            print('正在处理第 %d 行…………\n' % i)
        else:
            break
    print('处理完成 !!!')
    input_file.close()
    output_file.close()
    
segment2pair("subtitle.corpus","segment_result2.pair")