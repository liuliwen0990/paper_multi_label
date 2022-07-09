import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import pandas as pd
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_curve
from sklearn import metrics
import xlwt
#import matplotlib.pyplot as plt
import math
from pytorchtools import EarlyStopping
import time
from numpy.core.fromnumeric import argmax

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5 * math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def compute_rank(y_prob):
    rank = np.zeros(y_prob.shape)
    for i in range(len(y_prob)):
        temp = y_prob[i, :].argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(y_prob[i, :]))
        rank[i, :] = ranks
    return y_prob.shape[1] - rank

def compute_ranking_loss(y_prob, label):
    # y_predict = y_prob > 0.5
    num_samples, num_labels = label.shape
    loss = 0
    for i in range(num_samples):
        prob_positive = y_prob[i, label[i, :] > 0.5]
        prob_negative = y_prob[i, label[i, :] < 0.5]
        s = 0
        for j in range(prob_positive.shape[0]):
            for k in range(prob_negative.shape[0]):
                if prob_negative[k] >= prob_positive[j]:
                    s += 1

        label_positive = np.sum(label[i, :] > 0.5)
        label_negative = np.sum(label[i, :] < 0.5)
        if label_negative != 0 and label_positive != 0:
            loss = loss + s * 1.0 / (label_negative * label_positive)

    return loss * 1.0 / num_samples

def compute_one_error(y_prob, label):
    num_samples, num_labels = label.shape
    loss = 0
    for i in range(num_samples):
        pos = np.argmax(y_prob[i, :])
        loss += label[i, pos] < 0.5
    return loss * 1.0 / num_samples

def compute_coverage(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    coverage = 0
    for i in range(num_samples):
        if sum(label[i, :] > 0.5) > 0:
            coverage += max(rank[i, label[i, :] > 0.5])
    coverage = coverage * 1.0 / num_samples - 1
    return coverage / num_labels

def compute_average_precision(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    precision = 0
    for i in range(num_samples):
        positive = np.sum(label[i, :] > 0.5)
        rank_i = rank[i, label[i, :] > 0.5]
        temp = rank_i.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(rank_i))
        ranks = ranks + 1
        ans = ranks * 1.0 / rank_i
        if positive > 0:
            precision += np.sum(ans) * 1.0 / positive
    return precision / num_samples

def compute_macro_auc(y_prob, label):
    n, m = label.shape
    macro_auc = 0
    valid_labels = 0
    for i in range(m):
        if np.unique(label[:, i]).shape[0] == 2:
            index = np.argsort(y_prob[:, i])
            pred = y_prob[:, i][index]
            y = label[:, i][index] + 1
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
            temp = metrics.auc(fpr, tpr)
            macro_auc += temp
            valid_labels += 1
    macro_auc /= valid_labels
    return macro_auc

def label_transform(label):
    num_example, num_label = label.shape
    label = label.tolist()
    label_new = np.zeros((num_example,), dtype=np.int)
    # print(label)
    for i in range(0, num_example):
        label_new[i] = label[i].index(1.0)
    # print(label_new)
    return label_new

def label_predict(output,threshold = 0.5):
    #print(output)
    num_examples,num_labels = output.shape
    predict_label = np.zeros((num_examples,num_labels))
    for i in range(0,num_examples):
        for j in range(0,num_labels):
            if output[i,j]>=threshold:
                predict_label[i,j]=1
        if not 1 in output[i,:]:
            index = np.where(output[i,:]==max(output[i,:]))
            predict_label[i,index]=1

    return predict_label

def label_predict_new(output,thresholds):
    num_examples,num_labels = output.shape
    predict_label = np.zeros((num_examples,num_labels))
    for i in range(0,num_examples):
        for j in range(0,num_labels):
            if output[i,j]>=thresholds[j]:
                predict_label[i,j]=1
        if 1 not in output[i,:]:
            index = np.where(output[i,:]==max(output[i,:]))
            predict_label[i,index]=1
    return predict_label

def false_labels(true_label,predict_label):
    num_examples, num_labels = predict_label.shape
    false_label = 0
    for i in range(0,num_examples):
        for j in range(0,num_labels):
            if true_label[i,j]-predict_label[i,j]!=0:
                false_label = false_label+1
    #hamming_loss = false_labels/(num_labels*num_examples)
    return false_label

def compute_macro_f1(pred_label, label):
    up = np.sum(pred_label * label, axis=0)
    down = np.sum(pred_label, axis=0) + np.sum(label, axis=0)
    if np.sum(np.sum(label, axis=0) == 0) > 0:
        up[down == 0] = 0
        down[down == 0] = 1
    macro_f1 = 2.0 * np.sum(up / down)
    macro_f1 = macro_f1 * 1.0 / label.shape[1]
    return macro_f1

def get_train_data(filename_train,train_num_each = 40,mix_num = 7):
    data_train = pd.read_excel(filename_train)
    data_train = data_train.values
    #print('data type', type(data_train))
    data_train = np.around(data_train,6)
    x_train = data_train[:, 7:]
    y_train = data_train[:, 0:7]

    if mix_num ==1:
        x_train = x_train[:7*train_num_each,:]
        y_train = y_train[:7*train_num_each,:]
    elif mix_num == 2:
        x_train = x_train[:28 * train_num_each, :]
        y_train = y_train[:28 * train_num_each, :]
    elif mix_num == 3:
        x_train = x_train[:63 * train_num_each, :]
        y_train = y_train[:63 * train_num_each, :]
    elif mix_num == 4:
        x_train = x_train[:98 * train_num_each, :]
        y_train = y_train[:98 * train_num_each, :]
    elif mix_num == 5:
        x_train = x_train[:119 * train_num_each, :]
        y_train = y_train[:119 * train_num_each, :]
    elif mix_num == 6:
        x_train = x_train[:126 * train_num_each, :]
        y_train = y_train[:126 * train_num_each, :]
    x_train = x_train.reshape(x_train.shape[0], 1, 2, -1)
    return x_train, y_train

def get_test_data(filename_test):
    data_test = pd.read_excel(filename_test)
    data_test = data_test.values
    data_test = np.around(data_test, 6)

    x_test = data_test[:, 7:]
    y_test = data_test[:, 0:7]
    x_test = x_test.reshape(x_test.shape[0],1,  2, -1)
    return x_test, y_test

class CLDNN(nn.Module):

    def __init__(self):
        super(CLDNN, self).__init__()
        # in_channels输入数据通道数
        # padding:第一个数是高度，第二个是宽度
        dr = 0.6
        #self.n_features = 119*2
        self.n_features = 244*2 # number of parallel inputs
        #self.seq_len = seq_length  # number of timesteps
        self.n_hidden = 50  # number of hidden states
        self.n_layers = 1  # number of LSTM layers (stacked)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(1, 8), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(50),
            nn.ReLU(),
        )
        self.dropout1 = nn.Dropout(dr)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 8), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(50),
            nn.ReLU(),
        )
        self.dropout2 = nn.Dropout(dr)
        # fully connected layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 8), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(50),
            nn.ReLU(),
        )
        self.dropout3 = nn.Dropout(dr)
        self.lstm = nn.LSTM(input_size=self.n_features,hidden_size=self.n_hidden,num_layers=self.n_layers,batch_first= True)

        self.linear1 =  nn.Sequential(
            nn.Linear(50*50,256),
            nn.ReLU(),
        )

        #self.linear1 = nn.Linear(self.n_hidden*50,256)
        self.dropout4 = nn.Dropout(dr)
        self.linear2 = nn.Linear(256,7)

        #self.hidden = (torch.randn(1, 100, self.n_hidden), torch.randn(1, 100, self.n_hidden))

        #self.mlp1 = nn.Linear(50 * 122, 256)
        #nn.ReLU()
        #self.mlp2 = nn.Linear(256, 5)

    def forward(self, x):
        """
        input: N * 3 * image_size * image_size
        output: N * num_classes
        """
        batch_size = x.shape[0]
        x = self.conv1(x)
        x1 = self.dropout1(x)
        x = self.conv2(x1)
        x2 = self.dropout2(x)
        x = self.conv3(x2)
        x3 = self.dropout3(x) # (none,50,2,119)
        concat = torch.cat((x1,x3),3)#(none,50,2 244)
        #concat = x3

        concat_size = list(np.shape(concat))
        input_dim = int(concat_size[-1] * concat_size[-2])

        timesteps = int(concat_size[-3])
        x = concat.reshape([-1, timesteps,input_dim])
        x = x.to(device)
        #print(x.shape)
        #x = x.reshape([1,11200])

        '''
        hidden_state = torch.randn(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.randn(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)
        '''
        self.hidden = (torch.randn(self.n_layers, batch_size, self.n_hidden).to(device), torch.randn(self.n_layers, batch_size, self.n_hidden).to(device))
        #self.hidden = self.hidden.to(device)
        lstm_out, (h,c) = self.lstm(x,self.hidden)
        #print(lstm_out.shape)

        lstm_out = lstm_out.reshape([batch_size,-1])
        lstm_out = lstm_out.to(device)
        #x = self.mlp1(lstm_out)
        x = self.linear1(lstm_out)

        x = self.dropout4(x)
        x = self.linear2(x)

        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)

        #x = self.mlp1(x.view(x.size(0), -1))
        #x = self.mlp2(x)
        return x

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

def threshold_desicion_paper(train_pre_label,true_label):
    thresholds = []
    num_label = true_label.shape[1]
    train_sum = [sum(x) for x in zip(*train_pre_label)]
    true_sum = [sum(x) for x in zip(*true_label)]
    pre_multiply_true = np.multiply(train_pre_label,true_label)
    pre_multiply_true_sum = [sum(x) for x in zip(*pre_multiply_true)]

    for i in range(0,num_label):

        threshold =2*pre_multiply_true_sum[i]/(train_sum[i]+true_sum[i])
        thresholds.append(threshold)

    return thresholds

def threshold_desicion(pred_prob,true_label,mix):
    thresholds = [0]*true_label.shape[1]
    '''
    workbook = xlwt.Workbook(encoding='utf-8')
    recall_sheet = workbook.add_sheet('recall')
    precision_sheet = workbook.add_sheet('precision')
    '''
    for i in range(true_label.shape[1]):
        precision,recall,threshold = precision_recall_curve(true_label[:,i],pred_prob[:,i])
        target = precision + recall
        '''
        for j in range(len(precision)):
            recall_sheet.write(j,i,recall[j])
            precision_sheet.write(j,i,precision[j])
        '''
        index = argmax(target)
        thresholds[i] = threshold[index]
        print(thresholds[i])
    '''
    filename = 'train_recall_precision_mix_{}.xls'.format(mix)
    workbook.save(filename)
    '''
    return thresholds
snr_val = ['-18','-15','-12','-9','-6','-3']
print(torch.cuda.is_available())
#snr_val = ['-15', '-10', '-5', '0', '5', '10', '15']
#snr_val = ['-6']
train_num = '100'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
mix = 7
num_experiment = 10
for SNR in snr_val:
    row0 = ['hamming_loss', 'one-error', 'coverage', 'ranking_loss', 'average presicion', 'Macro-AUC', 'train_time',
            'SNR']
    workbook = xlwt.Workbook(encoding='utf-8')
    performance_sheet = workbook.add_sheet('result')
    for i in range(0, len(row0)):
        performance_sheet.write(0, i, row0[i])
    for l in range(num_experiment):
        print(SNR, train_num)
        model = CLDNN()
        model.apply(weight_init)
        model = model.to(device)
        BATCH_SIZE = 8
        train_filename = '0628_compound_snr_{}_train_{}.xlsx'.format(SNR, train_num)
        # test_filename = '0322_compound_snr_{}_test.xlsx'.format(SNR)
        X_train, Y_train = get_train_data(train_filename, train_num_each=int(train_num), mix_num=mix)
        num_train = X_train.shape[0]
        num_labels = Y_train.shape[1]

        X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
        Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)
        #num_experiment = 1  # 实验次数
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)
        train_set = Data.TensorDataset(X_train, Y_train)
        # 数据获取与处理
        m = nn.Sigmoid()
        loss_func = nn.BCELoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        # loss_count = []
        # macro_auc_count = []
        #hamming_loss_count = []
        #train_losses = []
        valid_losses = []
        #valid_hammingloss = []
        #avg_train_losses = []
        #avg_valid_losses = []

        n_epoch = 150
        patience = 10 # 当验证集损失在连续15次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
        early_stopping = EarlyStopping(patience, verbose=True)
        train_size = int(0.7 * len(train_set))
        valid_size = len(train_set) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_set, [train_size, valid_size])
        #train_out_prob = []
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True)
        valid_loader = Data.DataLoader(
            dataset=valid_dataset,
            shuffle=True)
        model.train()
        start = time.time()
        for epoch in range(1, n_epoch + 1):
            print('epoch=', epoch)
            model.train()
            for i, (x, y) in enumerate(train_loader):  # i:batch,x:data;y:target
                batch_x = Variable(x) # torch.Size([128, 1, 28, 28])
                batch_y = Variable(y)  # torch.Size([128])

                # 获取最后输出
                out = model(batch_x)  # torch.Size([128,10])

                out2 = m(out)
                # train_out_prob.append(out2)
                # print(type(out2))
                # 获取损失
                loss = loss_func(out2, batch_y.float())
                # print(type(loss)):<class 'torch.Tensor'>
                # 使用优化器优化损失
                opt.zero_grad()  # 清空上一步残余更新参数值
                loss.backward()  # 误差反向传播，计算参数更新值
                opt.step()  # 将参数更新值施加到net的parmeters上
                #train_losses.append(loss)
                # if i%5 == 0:
                # loss_count.append(loss)
                # print('{}:\t'.format(i), loss.item())
                # torch.save(model,'D:/Liuliwen/MLDF')

            model.eval()  # 设置模型为评估/测试模式
            falselabels = 0
            num_have_valided = 0
            predict_prob_valid = np.zeros((valid_size, num_labels))
            label_true_valid = np.zeros((valid_size, num_labels))
            pre_label_valid = np.zeros((valid_size, num_labels))
            for i, (data, target) in enumerate(valid_loader):
                # print(data)
                f = num_have_valided
                data = Variable(data)
                target = Variable(target)
                output = model(data)
                valid_probability = m(output)
                num_have_valided = f + len(target)
                valid_loss = loss_func(valid_probability, target.float())
                valid_losses.append(valid_loss)
                #predict_label_valid = label_predict(valid_probability)
                #falselabels = falselabels + false_labels(target, predict_label_valid)
                # print (falselabels)
                label_true_valid[f:num_have_valided, :] = target.cpu().numpy()
                #pre_label_valid[f:num_have_valided, :] = predict_label_valid
                predict_prob_valid[f:num_have_valided, :] = valid_probability.detach().cpu().numpy()
                # print(type(train_losses))
            #train_loss = torch.mean(torch.stack(train_losses))
            valid_loss = torch.mean(torch.stack(valid_losses))
            #hamming_loss_valid = falselabels / (valid_size * 5)
            #avg_train_losses.append(train_loss)
            #avg_valid_losses.append(valid_loss)
            #valid_hammingloss.append(hamming_loss_valid)
            #train_losses = []
            valid_losses = []

            early_stopping(valid_loss, model)
            # print(valid_hammingloss)
            #early_stopping(hamming_loss_valid, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        end = time.time()
        train_time = end - start
        print("训练结束\n epoch次数：%d" % (epoch))
        print("训练时间：%s Seconds" % (train_time))



        print('testing...')
        model.eval()
        test_filename = '0628_compound_snr_{}_test.xlsx'.format(SNR)
        x_test, y_test = get_test_data(test_filename)


        falselabels = 0
        num_have_tested = 0
        num_test = len(x_test)
        x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
        y_test = torch.from_numpy(y_test).type(torch.LongTensor)
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        test_set = Data.TensorDataset(x_test, y_test)
        test_loader = Data.DataLoader(
            dataset=test_set,
            batch_size=BATCH_SIZE,
            shuffle=True)
        predict_prob = np.zeros((num_test, num_labels))
        label_true = np.zeros((num_test, num_labels))
        pre_label = np.zeros((num_test, num_labels))
        model.eval()
        for x, y in test_loader:
            # print(len(y))
            test_x = Variable(x)
            test_y = Variable(y)
            # print(test_y.shape)
            a = num_have_tested
            num_have_tested = num_have_tested + len(test_y)
            out = model(test_x)
            out_probability = m(out)
            # 得到预测标签
            out_probability = out_probability.detach().cpu().numpy()
            #predict_label = label_predict_new(out_probability,thresholds)
            predict_label = label_predict(out_probability)
            falselabels = falselabels + false_labels(test_y, predict_label)
            # print (falselabels)
            label_true[a:num_have_tested, :] = y.cpu().numpy()
            pre_label[a:num_have_tested, :] = predict_label
            predict_prob[a:num_have_tested, :] = out_probability



        hamming_loss = falselabels / (num_test * 7)
        one_error = compute_one_error(predict_prob, label_true)
        coverage = compute_coverage(predict_prob, label_true)
        ranking_loss = compute_ranking_loss(predict_prob, label_true)
        average_precision = compute_average_precision(predict_prob, label_true)
        macro_auc = compute_macro_auc(predict_prob, label_true)

        print('Final test results:\t')
        print('hamming loss:\t', hamming_loss)
        print('one-error:\t', one_error)
        print('coverage:\t', coverage)
        print('ranking_loss:\t', ranking_loss)
        print('average precision:\t', average_precision)
        print('Macro-AUC:\t', macro_auc)
        performance_sheet.write(l + 1, 0, hamming_loss)
        performance_sheet.write(l + 1, 1, one_error)
        performance_sheet.write(l + 1, 2, coverage)
        performance_sheet.write(l + 1, 3, ranking_loss)
        performance_sheet.write(l + 1, 4, average_precision)
        performance_sheet.write(l + 1, 5, macro_auc)
        performance_sheet.write(l + 1, 6, train_time)
    filename = '0701_results_CLDNN_SNR_{}_trainnum_{}.xls'.format(SNR,train_num)
    workbook.save(filename)
'''
workbook2 = xlwt.Workbook(encoding = 'utf-8')
predict_probability_sheet = workbook2.add_sheet('probability')

for i in range(0,predict_prob.shape[0]):
    for j in range(0,num_labels):
        #print(out_probability[i,j])
        predict_probability_sheet.write(i, j, str(predict_prob[i,j]))
        predict_probability_sheet.write(i, j+5, label_true[i,j])
        predict_probability_sheet.write(i, j + 10, pre_label[i, j])
    #out_probability_sheet.write(i, 1, predict_label[i])
    #out_probability_sheet.write(i, 2, label_true[i])
filename2 = 'probability_{}_CLDNN_snr_{}.xls'.format(train_num,SNR)
workbook2.save(filename2)

'''












