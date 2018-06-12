import pickle
import logging
import os
import numpy as np
import csv

word_vector_dictctionary = {}

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

def init_seq(input_file,output_questionseq_file,output_answerseq_file):
    ###读取切好词的文本文件，加载全部词序列
    file_object = open(input_file, 'r', encoding='utf-8')
    file_len = load_len(input_file)
    vocab_dict = {}
    output_question_seq_file = csv.writer(open(output_questionseq_file,'w',encoding='utf-8',newline=''))
    output_answer_seq_file = csv.writer(open(output_answerseq_file, 'w',encoding='utf-8',newline=''))
    logging.info('开始加载词序列......')
    i = 1
    for i in range(2):
        print('正在加载词序列{0}/{1}......{2:.2f}%'.format(i+1,file_len,float(i/file_len)*100))
        question_seq = []
        answer_seq = []
        line = file_object.readline()
        if line:
            line_pair = line.split('|')
            line_question = line_pair[0].replace("\n",'').strip()
            print(line_question)
            line_answer = line_pair[1].replace("\n",'').strip()
            for word in line_question.split(' '):
                if word_vector_dictionary.__contains__(word):
                    question_seq.append(word_vector_dictionary[word])
                    print(question_seq)
            for word in line_answer.split(' '):
                if word_vector_dictionary.__contains__(word):
                    answer_seq.append(word_vector_dictionary[word])
                    print(answer_seq)
        #else:
            #break
        #question_seqs.append(question_seq)
        #answer_seqs.append(answer_seq)
        output_question_seq_file.writerow(question_seq)
        output_answer_seq_file.writerow(answer_seq)
        i = i + 1
    logging.info('词序列加载完毕 ! ! !')
    file_object.close()
    #output_question_seq_file.close()
    #output_answer_seq_file.close()

def load(question_vec_seq,answer_vec_seq):
    question_seqs = csv.reader(open(question_vec_seq,'r'))
    answer_seqs = csv.reader(open(answer_vec_seq, 'r'))
    return list(question_seqs),list(answer_seqs)

if __name__ == '__main__':
    word_vector_dictionary = load_vector('vectors.txt', 'word_vector_model2.txt')
    init_seq('./segment_result2.pair','question_vector_seq.csv','answer_vector_seq.csv')
    question_seqs,answer_seqs = load('question_vector_seq.csv','answer_vector_seq.csv')
    print('question_seqs[]:{0}'.format(len(question_seqs)))