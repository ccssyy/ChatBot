def load(inputfile):
    file_object = open(inputfile,'r',encoding='utf-8')
    print(len(file_object.read()))

filename = './segment_result2.pair'

if __name__ == '__main__':
    load(filename)