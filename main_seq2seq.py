# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 20:40:47 2018

@author: SamChen
"""

import sys
import math
import tflearn
import tensorflow as tf
#from tensorflow.python.ops import rnn_cell
#from tensorflow.python.ops import rnn
import chardet
import numpy as np
import os
import pickle
import logging
import csv
import struct

#import load_vector

#question_seqs = []
#answer_seqs = []

word_vec_dim = 200
max_seq_len = 8
word_vector_dict = {}
word_set = {}
word_set_path = './segment_result2.pair'
file_len = 128000
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_word_set(input):
    file_object = open(input, 'r', encoding='utf-8')
    logging.info('开始加载词集......')
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
    logging.info('词集加载完毕 ! ! !')
    file_object.close()
'''    
def load_vector(input,output):
    print('Begin Loading Vectors ......')
    
    input_file = open(input, "r", encoding='utf-8')
    output_file = open(output,"w", encoding='utf-8')
    
    #获取词表的数目及向量维度
    words_and_size = input_file.readline()
    words_and_size = words_and_size.strip()
    words = int(words_and_size.split(' ')[0])
    size = int(words_and_size.split(' ')[1])
    print('词数：%d' % words)
    print('维度：%d' % size)
    
    #word_vector = {}
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
        if word_set.__contains__(word):
            word_vector_dict[word] = vector
        
    input_file.close()
    
    #print(word_vector)
    print('Loading Vectors Finished !!!')
    output_file.write(str(word_vector_dict))
    output_file.close()
    
    return word_vector_dict
'''


def load_vector(input, output):
    logging.info('begin loading vectors ......')

    word_vector = {}
    if os.path.exists(output):
        print(output + " exist!")
        file = open(output, "rb")
        word_vector = pickle.load(file)
    else:
        print(output + " not exist!")
        input_file = open(input, "r", encoding='utf-8')
        output_file = open(output, "wb")

        # 获取词表的数目及向量维度
        words_and_size = input_file.readline()
        words_and_size = words_and_size.strip()
        words = int(words_and_size.split(' ')[0])
        size = int(words_and_size.split(' ')[1])
        print('词数：%d' % words)
        print('维度：%d' % size)

        #weight_str = ''

        for b in range(0, words):
            print('正在处理第 %d/%d 个词......%.2f%%' % (b + 1, words, float(((b + 1) / words) * 100)))

            line = input_file.readline()
            line = line.strip()
            word = line.split(' ', 1)[0]
            line2 = line.split(' ', 1)[1]
            vector = np.empty([200])
            # print(word)
            # print(line2)
            # print(line2.split(' '))
            for index in range(0, size):
                weight_str = line2.split(' ')[index]
                weight = float(weight_str)
                # print(weight)
                vector[index] = weight
            # print(vector)

            # 将词与对应向量存到dict中
            word_vector[word] = vector

        # print(word_vector)
        # output_file.write(str(word_vector))
        pickle.dump(word_vector, output_file)
        output_file.close()
        input_file.close()
    logging.info('loading vectors finished !!!')

    return word_vector

def load_len(inputfile):
    file_object = open(inputfile,'r',encoding='utf-8')
    file_len = len(file_object.read())
    file_object.close()
    return file_len

def load_seqs(question_vec_seq,answer_vec_seq):
    question_seqs = []
    answer_seqs = []
    with open(question_vec_seq,'r') as questions:
        with open(answer_vec_seq,'r') as answers:
            question_reader = csv.reader(questions)
            answer_reader = csv.reader(answers)
            for i,row in enumerate(question_reader):
                if i < 100000:
                    question_seqs.append(row)
                else:
                    break
            for i,row in enumerate(answer_reader):
                if i < 100000:
                    answer_seqs.append(row)
                else:
                    break
    return question_seqs,answer_seqs

def init_seq(input_file,output_questionseq_file,output_answerseq_file):
    ###读取切好词的文本文件，加载全部词序列
    file_object = open(input_file, 'r', encoding='utf-8')
    if os.path.exists(output_questionseq_file) and os.path.exists(output_answerseq_file):
        question_seqs,answer_seqs = load_seqs(output_questionseq_file,output_answerseq_file)
    else:
        #file_len = load_len(input_file)
        word_vector_dict = load_vector('vectors.txt', 'word_vector_model2.txt')
        output_question_seq_file = csv.writer(open(output_questionseq_file, 'w', encoding='utf-8', newline=''))
        output_answer_seq_file = csv.writer(open(output_answerseq_file, 'w', encoding='utf-8', newline=''))
        logging.info('开始加载词序列......')
        i = 1
        while i<=100000:
            print('正在加载词序列{0}/{1}......{2:.2f}%'.format(i+1,file_len,float(i/file_len)*100))
            question_seq = []
            answer_seq = []
            line = file_object.readline()
            if line:
                line_pair = line.split('|')
                line_question = line_pair[0].replace("\n",'').strip()
                line_answer = line_pair[1].replace("\n",'').strip()
                for word in line_question.split(' '):
                    if word_vector_dict.__contains__(word):
                        question_seq.append(word_vector_dict[word])
                for word in line_answer.split(' '):
                    if word_vector_dict.__contains__(word):
                        answer_seq.append(word_vector_dict[word])
            else:
                break
            #question_seqs.append(question_seq)
            #answer_seqs.append(answer_seq)
            output_question_seq_file.writerow(question_seq)
            output_answer_seq_file.writerow(answer_seq)
            i = i + 1
        logging.info('词序列加载完毕 ! ! !')
        question_seqs, answer_seqs = load_seqs(output_questionseq_file, output_answerseq_file)
        file_object.close()
    return question_seqs,answer_seqs
    
def vector_sqrtlen(vector):
    len = 0
    for item in vector:
        len += item * item
    len = math.sqrt(len)
    return len

def vector_cosine(v1, v2):
    if len(v1) != len(v2):
        sys.exit(1)
    sqrtlen1 = vector_sqrtlen(v1)
    sqrtlen2 = vector_sqrtlen(v2)
    value = 0
    for item1, item2 in zip(v1, v2):
        value += item1 * item2
    return value / (sqrtlen1*sqrtlen2)

def vector2word(vector):
    max_cos = -10000
    for word in word_vector_dict:
        v = word_vector_dict[word]
        cosine = vector_cosine(vector, v)
        if cosine > max_cos:
            max_cos = cosine
            match_word = word
    return (match_word, max_cos)

class MainSeq2Seq(object):
    """
    思路：输入输出序列一起作为input，然后通过slick和unstack切分
    完全按照编码器译码器来做
    输出的时候把解码器的输出按照词向量的200维展平，这样输出就是(?,seqlen*200)
    这样就可以通过regression来做回归计算了，输入的y 也展平，保持一致
    """
    def __init__(self, max_seq_len = 16, word_vec_dim = 200, input_file = word_set_path):
        self.max_seq_len = max_seq_len
        self.word_vec_dim = word_vec_dim
        self.input_file = input_file
        
    def generate_training_data(self,question_seqs,answer_seqs,startpoint = 0,size = 128):
        #load_word_set(self.input_file)
        #word_vector_dict = load_vector('vectors.txt','word_vector_model2.txt')
        #question_seqs,answer_seqs = init_seq(self.input_file,'question_vector_seq.csv','answer_vector_seq.csv')
        xy_data = []
        y_data = []
        logging.info('正在生成训练数据......')
        for i in range(size):
            print('正在处理第 %d/%d 个序列......%.2f%%' % (i,len(question_seqs),float(i/len(question_seqs))*100))
            question_seq = question_seqs[startpoint]
            answer_seq = answer_seqs[startpoint]
            if len(question_seq) < self.max_seq_len and len(answer_seq) < self.max_seq_len:
                sequence_xy = [np.zeros(self.word_vec_dim)] * (self.max_seq_len-len(question_seq)) + list(reversed(question_seq))
                sequence_y = answer_seq + [np.zeros(self.word_vec_dim)] * (self.max_seq_len-len(answer_seq))
                sequence_xy = sequence_xy + sequence_y
                sequence_y = [np.ones(self.word_vec_dim)] + sequence_y
                xy_data.append(sequence_xy)
                y_data.append(sequence_y)
        logging.info('训练数据生成完成 ! ! !')
        return np.array(xy_data), np.array(y_data)
    def model(self, feed_previous=False):
        # 通过输入的XY生成encoder_inputs和带GO头的decoder_inputs
        logging.info('正在生成encoder_inputs和带GO头的decoder_inputs......')
        input_data = tflearn.input_data(shape=[None, self.max_seq_len*2, self.word_vec_dim], dtype=tf.float32,name = "XY")
        encoder_inputs = tf.slice(input_data, [0,0,0], [-1,self.max_seq_len,self.word_vec_dim], name = "enc_in")
        decoder_inputs_tmp = tf.slice(input_data, [0,self.max_seq_len,0], [-1,self.max_seq_len-1,self.word_vec_dim], name = "dec_in_tmp")
        go_inputs = tf.ones_like(decoder_inputs_tmp)
        go_inputs = tf.slice(go_inputs, [0,0,0], [-1,1,self.word_vec_dim])
        decoder_inputs = tf.concat([go_inputs, decoder_inputs_tmp], 1, name = "dec_in")
        
        #编码器
        #把encoder_inputs交个编码器，返回一个输出(预测序列的第一个值)和一个状态(传给解码器)
        logging.info('生成编码器......')
        (encoder_output_tensor, states) = tflearn.lstm(encoder_inputs, self.word_vec_dim, return_state = True, scope = 'encoder_lstm')
        encoder_output_sequence = tf.stack([encoder_output_tensor], axis=1)
        
        #解码器
        #预测过程前一个时间序的输出作为下一个时间序的输入
        #先用编码器的最后一个输出作为第一个输入
        logging.info('生成解码器......')
        if feed_previous:
            first_dec_input = go_inputs
        else:
            first_dec_input = tf.slice(decoder_inputs, [0,0,0], [-1,1,self.word_vec_dim])
        decoder_output_tensor = tflearn.lstm(first_dec_input, self.word_vec_dim, initial_state=states, return_seq=False, reuse=False, scope = 'decoder_lstm')
        decoder_output_sequence_single = tf.stack([decoder_output_tensor], axis=1)
        decoder_output_sequence_list = [decoder_output_tensor]
        #再用解码器的输出作为下一个时序的输入
        for i in range(self.max_seq_len-1):
            print('正在用解码器的输出作为下一个时序的输入......%.2f%%' % float(i/(self.max_seq_len-1)))
            if feed_previous:
                next_dec_input = decoder_output_sequence_single
            else:
                next_dec_input = tf.slice(decoder_inputs, [0,i+1,0], [-1,1,self.word_vec_dim])
            decoder_output_tensor = tflearn.lstm(next_dec_input, self.word_vec_dim, return_state=False, reuse=True, scope = 'decoder_lstm')
            decoder_output_sequence_single = tf.stack([decoder_output_tensor], axis=1)
            decoder_output_sequence_list.append(decoder_output_tensor)
        logging.info('解码器输出序列表生成完成 ! ! !')
        
        decoder_output_sequence = tf.stack(decoder_output_sequence_list, axis=1)
        real_output_sequence = tf.concat([encoder_output_sequence, decoder_output_sequence], 1)
        
        net = tflearn.regression(real_output_sequence, optimizer='adam', learning_rate=0.001, loss='mean_square')
        model = tflearn.DNN(net)
        return model
    
    def train(self):
        load_word_set(self.input_file)
        # word_vector_dict = load_vector('vectors.txt','word_vector_model2.txt')
        question_seqs, answer_seqs = init_seq(self.input_file, 'question_vector_seq.csv', 'answer_vector_seq.csv')
        model = self.model(feed_previous=False)
        for index in range(int(1/2*load_len(file_len)/128)):
            logging.info('index: {0}'.format(index))
            if os.path.exists('./model/model'):
                model = model.load('./model/model')
            trainXY,trainY = self.generate_training_data(question_seqs, answer_seqs,startpoint=index)
            #model = self.model(feed_previous=False)
            model.fit(trainXY, trainY, n_epoch=100, snapshot_epoch=False, batch_size=1)
            logging.info('正在保存模型......')
            model.save('./model/model')
            logging.info('模型保存完成 ! ! !')
        return model
    
    def load(self):
        model = self.model(feed_previous=True)
        print('正在加载模型......')
        model.load('./model/model')
        return model
    
if __name__ == '__main__':
    #print(sys.argv)
    main_seq2seq = MainSeq2Seq(word_vec_dim=word_vec_dim, max_seq_len=max_seq_len)
    main_seq2seq.train()
    '''
    phrase = sys.argv[1]
    if 3 == len(sys.argv):
        main_seq2seq = MainSeq2Seq(word_vec_dim=word_vec_dim, max_seq_len=max_seq_len, input_file=sys.argv[2])
    else:
        main_seq2seq = MainSeq2Seq(word_vec_dim=word_vec_dim, max_seq_len=max_seq_len)
    if phrase == 'train':
        main_seq2seq.train()
    else:
        model = main_seq2seq.load()
        trainXY, trainY = main_seq2seq.generate_training_data()
        predict = model.predict(trainXY)
        for sample in predict:
            print('predict answer:')
            for w in sample[1:]:
                (match_word, max_cos) = vector2word(w)
                print(match_word, max_cos, vector_sqrtlen(w))
    '''