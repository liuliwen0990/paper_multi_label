import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import pandas as pd
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
import xlwt
import matplotlib.pyplot as plt
import math
from pytorchtools import EarlyStopping
import time

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

    return predict_label

def label_predict_new(output,thresholds):
    num_examples,num_labels = output.shape
    predict_label = np.zeros((num_examples,num_labels))
    for i in range(0,num_examples):
        for j in range(0,num_labels):
            if output[i,j]>=thresholds[j]:
                predict_label[i,j]=1

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

def get_data(filename_train, filename_test,train_num_each = 40,mix_num = 5):
    data_train = pd.read_excel(filename_train)
    data_train = data_train.values
    data_test = pd.read_excel(filename_test)
    data_test = data_test.values
    x_train = data_train[:, 5:]
    y_train = data_train[:, 0:5]
    x_test = data_test[:, 5:]
    x_test_I = data_test[:, 5:133]
    x_test_Q = np.zeros((len(x_test),128))
    x_test_new = np.hstack((x_test_I,x_test_Q))
    y_test = data_test[:, 0:5]

    x_train = x_train.reshape(x_train.shape[0],  2, -1)
    x_test = x_test.reshape(x_test.shape[0],  2, -1)
    x_test_new = x_test_new.reshape(x_test_new.shape[0],  2, -1)
    if mix_num ==1:
        x_train = x_train[:5*train_num_each,:]
        y_train = y_train[:5*train_num_each,:]
    elif mix_num == 2:
        x_train = x_train[:15 * train_num_each, :]
        y_train = y_train[:15 * train_num_each, :]
    elif mix_num == 3:
        x_train = x_train[:25 * train_num_each, :]
        y_train = y_train[:25 * train_num_each, :]
    elif mix_num == 4:
        x_train = x_train[:30 * train_num_each, :]
        y_train = y_train[:30 * train_num_each, :]
    #y_train = label_transform(y_train)
    #y_test = label_transform(y_test)
    return x_train, y_train, x_test, y_test, x_test_new


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

def threshold_desicion(train_pre_label,true_label):
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


class Residual_Unit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.batch_norm_1 = nn.BatchNorm1d(num_features=32)
        self.batch_norm_2 = nn.BatchNorm1d(num_features=32)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu(output)
        output = self.batch_norm_1(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.batch_norm_2(output)

        output = output + input
        return output


class Residual_Stack(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(num_features=32)
        self.residual_unit_1 = Residual_Unit()
        self.residual_unit_2 = Residual_Unit()
        self.max_pool = nn.MaxPool1d(2)

    def forward(self, input):
        output = self.conv1(input)
        output = self.batch_norm(output)
        output = self.residual_unit_1(output)
        output = self.residual_unit_2(output)
        output = self.max_pool(output)
        return output


class RESNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_stack_1 = Residual_Stack(in_channels=2)
        self.residual_stack_2 = Residual_Stack()
        self.residual_stack_3 = Residual_Stack()
        self.residual_stack_4 = Residual_Stack()
        self.residual_stack_5 = Residual_Stack()

        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 5)

        self.dropout = nn.AlphaDropout(0.1)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output = self.residual_stack_1(input)
        output = self.residual_stack_2(output)
        output = self.residual_stack_3(output)
        output = self.residual_stack_4(output)
        output = self.residual_stack_5(output)

        output = output.view(-1, 128)

        output = self.linear1(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.linear2(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.linear3(output)
        #output = self.softmax(output)

        return output
#snr_val = ['0','-3','-6','-9','-12','-15','-18']
snr_val = ['-12','-15','-18']
#snr_val = ['0']
train_val = ['100']
#mix = 4
for mix in range(5,6):
    for kk in range(0,len(snr_val)):
        for mm in range(0,len(train_val)):
            train_num = train_val[mm]
            SNR = snr_val[kk]

            BATCH_SIZE = 64
            num_experiment = 10 #实验次数
            train_filename = 'compound_{}_snr_{}_train_new.xlsx'.format(train_num,SNR)
            test_filename = 'compound_{}_snr_{}_test_new.xlsx'.format(train_num,SNR)
            x_train, y_train, x_test, y_test, x_test_I = get_data(train_filename,test_filename,train_num_each = int(train_num),mix_num = mix)
            num_train = len(x_train)
            num_test = len(x_test)
            num_labels = y_test.shape[1]
            x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
            y_train = torch.from_numpy(y_train).type(torch.LongTensor)
            x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
            y_test = torch.from_numpy(y_test).type(torch.LongTensor)
            x_test_I = torch.from_numpy(x_test_I).type(torch.FloatTensor)

            train_set = Data.TensorDataset(x_train, y_train)
            test_set = Data.TensorDataset(x_test, y_test)
            test_set_I = Data.TensorDataset(x_test_I, y_test)
            test_loader = Data.DataLoader(
                dataset=test_set,
                batch_size=BATCH_SIZE,
                shuffle=True)
            test_loader_I = Data.DataLoader(
                dataset=test_set_I,
                batch_size=BATCH_SIZE,
                shuffle=True)
            row0 = ['hamming_loss','one-error','coverage','ranking_loss','average presicion','Macro-AUC','train_time']
            workbook = xlwt.Workbook(encoding = 'utf-8')
            performance_sheet = workbook.add_sheet('result')
            for i in range(0, len(row0)):
                performance_sheet.write(0, i, row0[i])
            for l in range(num_experiment):
                print(SNR, train_num, mix)
                model = RESNET()
                #print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
                #model.apply(weight_init)
                # print(model)
                # 数据获取与处理
                m = nn.Sigmoid()
                loss_func = nn.BCELoss()
                opt = torch.optim.Adam(model.parameters(), lr=1e-4)
                #loss_count = []
                macro_auc_count = []
                hamming_loss_count = []
                train_losses = []
                valid_losses = []
                valid_hammingloss = []
                avg_train_losses = []
                avg_valid_losses = []

                n_epoch = 150
                patience = 15  # 当验证集损失在连续15次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
                print("patience:",patience)
                early_stopping = EarlyStopping(patience, verbose=True)
                train_size = int(0.8 * len(train_set))
                valid_size = len(train_set) - train_size
                train_dataset, valid_dataset = torch.utils.data.random_split(train_set, [train_size, valid_size])
                train_out_prob = []
                train_loader = Data.DataLoader(
                    dataset=train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True)
                valid_loader = Data.DataLoader(
                    dataset=valid_dataset,
                    shuffle=True)
                model.train()
                start = time.time()
                for epoch in range(1,n_epoch+1):

                    print('epoch=', epoch)
                    model.train()
                    for i, (x, y) in enumerate(train_loader):#i:batch,x:data;y:target
                        batch_x = Variable(x)  # torch.Size([128, 1, 28, 28])
                        batch_y = Variable(y)  # torch.Size([128])

                        # 获取最后输出
                        out = model(batch_x)  # torch.Size([128,10])
                        out2 = m(out)
                        #train_out_prob.append(out2)
                        #print(type(out2))
                        # 获取损失
                        loss = loss_func(out2, batch_y.float())
                        #print(type(loss)):<class 'torch.Tensor'>
                        # 使用优化器优化损失
                        opt.zero_grad()  # 清空上一步残余更新参数值
                        loss.backward()  # 误差反向传播，计算参数更新值
                        opt.step()  # 将参数更新值施加到net的parmeters上
                        train_losses.append(loss)
                        #if i%5 == 0:
                            #loss_count.append(loss)
                            #print('{}:\t'.format(i), loss.item())
                        # torch.save(model,'D:/Liuliwen/MLDF')

                    model.eval()  # 设置模型为评估/测试模式
                    falselabels = 0
                    num_have_valided = 0
                    predict_prob_valid = np.zeros((valid_size, num_labels))
                    label_true_valid = np.zeros((valid_size, num_labels))
                    pre_label_valid = np.zeros((valid_size, num_labels))
                    for i,(data, target) in enumerate(valid_loader):
                        #print(data)
                        f = num_have_valided
                        data = Variable(data)
                        target = Variable(target)
                        output = model(data)
                        valid_probability = m(output)
                        num_have_valided = f + len(target)
                        valid_loss = loss_func(valid_probability, target.float())
                        valid_losses.append(valid_loss)
                        predict_label_valid = label_predict(valid_probability)
                        falselabels = falselabels + false_labels(target, predict_label_valid)
                        # print (falselabels)
                        label_true_valid[f:num_have_valided, :] = target
                        pre_label_valid[f:num_have_valided, :] = predict_label_valid
                        predict_prob_valid[f:num_have_valided, :] = valid_probability.detach().numpy()
                    #print(type(train_losses))
                    train_loss = torch.mean(torch.stack(train_losses))
                    valid_loss = torch.mean(torch.stack(valid_losses))
                    hamming_loss_valid = falselabels / (valid_size * 5)
                    avg_train_losses.append(train_loss)
                    avg_valid_losses.append(valid_loss)
                    valid_hammingloss.append(hamming_loss_valid)
                    train_losses = []
                    valid_losses = []

                    early_stopping(valid_loss, model)
                    #print(valid_hammingloss)
                    #early_stopping(hamming_loss_valid, model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                end = time.time()
                train_time = end - start
                print("训练结束\n epoch次数：%d"% (epoch))
                print("训练时间：%s Seconds" % (train_time))
                model.eval()
                falselabels = 0
                num_have_tested = 0
                predict_prob = np.zeros((num_test, num_labels))
                label_true = np.zeros((num_test, num_labels))
                pre_label = np.zeros((num_test, num_labels))
                model.eval()
                for x, y in test_loader:
                    #print(len(y))
                    test_x = Variable(x)
                    test_y = Variable(y)
                    # print(test_y.shape)
                    a = num_have_tested
                    num_have_tested = num_have_tested + len(test_y)
                    out = model(test_x)
                    out_probability = m(out)
                    # 得到预测标签
                    predict_label = label_predict(out_probability)
                    falselabels = falselabels + false_labels(test_y, predict_label)
                    # print (falselabels)
                    label_true[a:num_have_tested, :] = y
                    pre_label[a:num_have_tested, :] = predict_label
                    predict_prob[a:num_have_tested, :] = out_probability.detach().numpy()

                hamming_loss = falselabels / (num_test * 5)
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
                performance_sheet.write(l+1,0,hamming_loss )
                performance_sheet.write(l+1,1,one_error)
                performance_sheet.write(l+1,2,coverage)
                performance_sheet.write(l+1,3,ranking_loss)
                performance_sheet.write(l+1,4,average_precision)
                performance_sheet.write(l+1,5,macro_auc)
                performance_sheet.write(l + 1, 6, train_time)
                #accuracy_partial_count(label_true, pre_label)
                #迁移学习，测试集只有一路信号

                '''
                falselabels = 0
                num_have_tested = 0
                predict_prob = np.zeros((num_test, num_labels))
                label_true = np.zeros((num_test, num_labels))
                pre_label = np.zeros((num_test, num_labels))
                model.eval()
                for x, y in test_loader_I:
                    # print(len(y))
                    test_x = Variable(x)
                    test_y = Variable(y)
                    # print(test_y.shape)
                    a = num_have_tested
                    num_have_tested = num_have_tested + len(test_y)
                    out = model(test_x)
                    out_probability = m(out)
                    # 得到预测标签
                    predict_label = label_predict(out_probability)
                    falselabels = falselabels + false_labels(test_y, predict_label)
                    # print (falselabels)
                    label_true[a:num_have_tested, :] = y
                    pre_label[a:num_have_tested, :] = predict_label
                    predict_prob[a:num_have_tested, :] = out_probability.detach().numpy()

                hamming_loss_I = falselabels / (num_test * 5)
                one_error_I = compute_one_error(predict_prob, label_true)
                coverage_I = compute_coverage(predict_prob, label_true)
                ranking_loss_I = compute_ranking_loss(predict_prob, label_true)
                average_precision_I = compute_average_precision(predict_prob, label_true)
                macro_auc_I = compute_macro_auc(predict_prob, label_true)
                print('Final test_I results:\t')
                print('hamming loss:\t', hamming_loss_I)
                print('one-error:\t', one_error_I)
                print('coverage:\t', coverage_I)
                print('ranking_loss:\t', ranking_loss_I)
                print('average precision:\t', average_precision_I)
                print('Macro-AUC:\t', macro_auc_I)
                performance_sheet.write(l + 2 +num_experiment, 0, hamming_loss_I)
                performance_sheet.write(l + 2+ num_experiment, 1, one_error_I)
                performance_sheet.write(l + 2+ num_experiment, 2, coverage_I)
                performance_sheet.write(l + 2+ num_experiment, 3, ranking_loss_I)
                performance_sheet.write(l + 2+ num_experiment, 4, average_precision_I)
                performance_sheet.write(l + 2+ num_experiment, 5, macro_auc_I)
                #accuracy_partial_count(label_true, pre_label)
                
                plt.figure('PyTorch_CNN_Loss')
                plt.plot(loss_count, label='Loss')
                plt.legend()
            
                plt.figure('Macro_AUC_test')
                plt.plot(macro_auc_count,label='macro_auc')
                plt.legend()
            
                plt.figure('Hamming_Loss_test')
                plt.plot(hamming_loss_count, label='hamming_loss')
                plt.legend()
                plt.show()
                '''

            filename = '0626_results_{}_RESNET_snr_{}_mix_{}.xls'.format(train_num,SNR,mix)
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












