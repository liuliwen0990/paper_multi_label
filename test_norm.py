from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from learner.cascade import Cascade
from learner.measure import *
import numpy as np
import pandas as pd
import xlsxwriter
import xlwt
import time
import random
def read_data(filename,num_label=5,mix_num = 5,train_num_each = 40):
    df = pd.read_excel(filename)
    #df = df.reset_index(drop=True)


    #print(filename,num_train)
    #signal_length  = width-num_label
    #signal_length_I = int((signal_length)/2)
    #num_label = 5
    data = df.values
    label = data[:,:num_label]
    #signal = data[:,num_label:(num_label+signal_length)]
    signal = data[:, num_label:]
    #单种信号实际位置：499，但要写成：500；1+2：1500；1+2+3：2500；1+2+3+4:3000；all:3100
    #种类：1:7；1+2:28；1+2+3:63；1+2+3+4:98；1+2+3+4+5:119;1+...+6:126;all:127
    if mix_num ==1:
        label = label[:7*train_num_each,:]
        signal = signal[:7*train_num_each,:]
    elif mix_num == 2:
        label = label[:28*train_num_each,:]
        signal = signal[:28*train_num_each,:]
    elif mix_num == 3:
        label = label[:63*train_num_each,:]
        signal = signal[:63*train_num_each,:]
    elif mix_num == 4:
        label = label[:98*train_num_each,:]
        signal = signal[:98*train_num_each,:]
    elif mix_num == 5:
        label = label[:119*train_num_each,:]
        signal = signal[:119*train_num_each,:]
    elif mix_num == 6:
        label = label[:126*train_num_each,:]
        signal = signal[:126*train_num_each,:]
    num_train = label.shape[0]
    '''
    len_signal = int((width-num_label)/2)
    data_test = np.empty([num_train,2,len_signal])
    for i in range(0,num_train):
        data_test[i,0,:]=data_single[i,:len_signal]
        data_test[i,1,:]=data_single[i,len_signal:]
    
    index = range(0, num_train)
    index = shuffle(index)
    label_test = label_test[index]
    data_test = data_test[index]
    return label_test,data_test
    '''
    index = range(0, num_train)
    index = shuffle(index)
    label = label[index]
    signal = signal[index]

    return label,signal

def norm_data(x_train,x_test):
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    x = np.vstack((x_train,x_test))
    x = scale(x)
    x = np.around(x,4)
    x_train_new = x[:n_train,:]
    x_test_new = x[n_train:,:]
    return x_train_new,x_test_new

def shuffle_index(num_samples):
    a = range(0, num_samples)
    a = shuffle(a)
    length = int((num_samples + 1) / 2)
    train_index = a[:length]
    test_index = a[length:]
    return [train_index, test_index]

# 划分训练集与测试集
def make_data(dataset):
    data = np.load("dataset/{}_data.npy".format(dataset))
    label = np.load("dataset/{}_label.npy".format(dataset))
    label = label.astype("int")

    num_samples = data.shape[0]
    train_index, test_index = shuffle_index(num_samples)
    train_data = data[train_index]
    train_label = label[train_index]
    test_data = data[test_index]
    test_label = label[test_index]
    return [train_data, train_label, test_data, test_label]
def accuracy_partial_count(true_label,predict_label):
    num_example,num_label = true_label.shape
    m = []
    for i in range(0,num_example):
        count = 0
        for j in range(0,num_label):
            if true_label[i,j] == predict_label[i,j]:
                count =count+1
        m.append(count)
    for threshold in range(1,num_label+1):
        true_count  = sum(i >= threshold for i in m)
        accuracy = true_count / num_example
        print('>=',threshold,'accuracy:',accuracy)
    return

if __name__ == '__main__':
    dataset = "signal"
    '''
    dataset = "image"
    data_train, label_train, data_test, label_test = make_data(dataset)
    print(label_train[0])
    '''
    num_experiment = 2
    #snr_val = ['-18','-15','-12','-9','-6','-3','0']
    snr_val = [ '-9']
    train_num = '20'
    mix = 7
    #supervisor = "average precision"
    #supervisor = "ranking loss"
    #supervisor = "one-error"
    #supervisor = "coverage"
    #supervisor = "hamming loss"
    supervisor ="macro_auc"
    for snr in snr_val:
        print (supervisor)

        print(snr, train_num, mix)


        row0 = ['hamming_loss', 'one-error', 'coverage', 'ranking_loss', 'average presicion', 'Macro-AUC']
        workbook = xlwt.Workbook(encoding='utf-8')
        performance_sheet = workbook.add_sheet('result')
        ra = random.randint(1,99)
        filename = 'results0628_MLDF_{}_{}_snr_{}_mix_{}_{}.xls'.format( train_num, supervisor, snr, mix,ra)

        for i in range(0, len(row0)):
            performance_sheet.write(0, i, row0[i])
        for l in range(num_experiment):

            label_train, data_train = read_data('0628_compound_snr_{}_train_{}.xlsx'.format(snr,train_num), num_label=7,
                                                mix_num=mix, train_num_each=int(train_num))
            label_test, data_test = read_data('0628_compound_snr_{}_test.xlsx'.format(snr), num_label=7, train_num_each = 20)
            num_train = len(label_train)
            num_test = len(label_test)

            index = range(0, num_train)
            index = shuffle(index)
            label_train = label_train[index]
            data_train = data_train[index]

            #data_train,data_test = norm_data(data_train,data_test)
            #print(data_train[0:10,:])

            model = Cascade(dataset, max_layer=80, num_forests=4, n_fold=4, step=3

                            )
            start = time.time()
            model.train(data_train, label_train, supervisor, n_estimators=100)
            end = time.time()
            train_time = end - start
            print("训练时间：%s Seconds" % (train_time))
            test_prob = model.predict(data_test, supervisor)
            threshold = 0.5
            #print(test_prob)
            print ("是否有缺省值：", np.isnan(test_prob).any())  # False没有
            test_prob[np.isnan(test_prob)] = 0
            value = do_metric(test_prob, label_test, threshold)#0.5 threshold

            print('test:I,Q:')
            print(["hamming loss", "one-error", "coverage", "ranking loss", "average precision", "macro-auc"])
            print(value)
            for j in range(len(value)):
                performance_sheet.write(l+1,j,value[j])
            performance_sheet.write(l+1,j+1,train_time)
            performance_sheet.write(l + 1, j + 2, supervisor)
            label_predict = test_prob > threshold
        workbook.save(filename)

'''
workbook = xlsxwriter.Workbook('predict_true.xlsx')    
data_sheet = workbook.add_worksheet('compare')
row,col = label_test.shape
row0=['bask','gfsk','wbfm','am','dsb']
data_sheet.write_row('A1',row0) 
for i in range(1,row+1):
    for j in range(0,col):
        data_sheet.write(i,j,label_test[i-1,j])
        data_sheet.write(i,j+col,label_predict[i-1,j])
        data_sheet.write(i,j+col*2,test_prob[i-1,j])
workbook.close()
'''
    

