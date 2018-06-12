# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 20:36:33 2018

@author: SamChen
"""

import jieba
from jieba import analyse

def segment(input, output):
    input_file = open(input, "r", encoding='utf-8')
    output_file = open(output, "w", encoding='utf-8')
    while True:
        line = input_file.readline()
        if line:
            line = line.strip()
            seg_list = jieba.cut(line)
            segments = ""
            for str in seg_list:
                segments = segments + " " + str
            output_file.write(segments)
        else:
            break
    input_file.close()
    output_file.close()
    
segment("subtitle.corpus","segment_result.txt")