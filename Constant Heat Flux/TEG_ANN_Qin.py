# Conventional Bulk TEG ANN training
# Data available on https://github.com/LorewalkerZYX/Bulk-TEG-project.git

import pandas as pd
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
import xlsxwriter
import time

time_cost = 0


# Set the random seed manually for reproducibility.
def seed_torch(seed=1029):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


seed_torch(10)
device = torch.device('cuda:0')


def LoadData():
    # preparing data
    data_X = pd.read_excel('input.xlsx')
    data_Y = pd.read_excel('Output.xlsx')
    dataX = data_X.iloc[:, :]

    dataY = data_Y.iloc[:, 4:5]
    X_train = dataX.to_numpy()
    Y_train = dataY.to_numpy()
    return X_train, Y_train


# const = 100000


# normalization
def normalize_x(x, input=True):
    temp = x

    if input:
        wn = 4.5  # wn = [0.5-5]
        wp = 4.5  # wp = [0.5-5]
        h = 4.5  # h = [0.5-5]
        h_ic = 2.5  # h_ic = [0.5-3]
        ff = 0.9  # ff = [0.05-0.95]
        q_in = 4000  # q_in = [1000-5000]
        rho_c = 9.9E-8  # rho_c = [1E-9-1E-7]

        for i in range(len(temp)):
            temp[i, 0] = (temp[i, 0] - 0.5) / wn
            temp[i, 1] = (temp[i, 1] - 0.5) / wp
            temp[i, 2] = (temp[i, 2] - 0.5) / h
            temp[i, 3] = (temp[i, 3] - 0.5) / h_ic
            temp[i, 4] = (temp[i, 4] - 0.05) / ff
            temp[i, 5] = (temp[i, 5] - 1000) / q_in
            temp[i, 6] = (temp[i, 6] - 1E-9) / rho_c

    else:
        for k in range(len(temp)):
            pass
            # temp[k, 0] *= const
            temp[k][0] = np.log(temp[k][0])
            temp[k][0] = (temp[k][0] - M) / S
    return temp


def recover_y(y):
    # y /= const
    y = y * S + M
    outy = np.exp(y)
    return outy


def train_dev_split(X, Y, dev_ratio=0.25):
    size = int(len(X) * (1 - dev_ratio))
    label = np.array(range(len(X)))
    SelectT = random.sample(range(len(X)), size)  # np.random.randint(0, len(X) - 1, size)
    train_x = X[SelectT]
    train_y = Y[SelectT]
    SelectV = np.delete(label, SelectT)
    valid_x = X[SelectV]
    valid_y = Y[SelectV]
    # print(len(label), len(SelectT), len(SelectV))
    # return X[:size], Y[:size], X[size:], Y[size:]
    return train_x, train_y, valid_x, valid_y


[X_train, Y_train] = LoadData()

dev_ratio = 0.1
Train_x, Train_y, test_x, test_y = train_dev_split(X_train, Y_train, dev_ratio)

trainx, trainy, validx, validy = train_dev_split(Train_x, Train_y, dev_ratio)

testing_x = test_x.copy()
testing_y = test_y.copy()

M = -5.713109082133402
S = 1.461078971097606

trainx = normalize_x(trainx)
validx = normalize_x(validx)
testx = normalize_x(test_x)
trainy = normalize_x(trainy, False)
validy = normalize_x(validy, False)
Y_test = normalize_x(test_y, False)


# dataset
train_size = trainx.shape[0]
valid_size = validx.shape[0]

Batch_size = 64
epoch = 2000
learning_rate = 0.002
hidden_feature = 20
LayerNumber = 2
n = 0
step = 1
# print(train_size, valid_size)

# trainsfer numpy to torch
x = torch.from_numpy(trainx)
x = x.type(torch.FloatTensor)

y = torch.from_numpy(trainy)
y = y.type(torch.FloatTensor)

X_dev = torch.from_numpy(validx)
X_dev = X_dev.type(torch.FloatTensor)

Y_dev = torch.from_numpy(validy)
Y_dev = Y_dev.type(torch.FloatTensor)

train_data = Data.TensorDataset(x, y)
val_data = Data.TensorDataset(X_dev, Y_dev)

X_test = torch.from_numpy(testx)
X_test = X_test.type(torch.FloatTensor)

'''Y_test = torch.from_numpy(Y_test)
Y_test = Y_test.type(torch.FloatTensor)'''

loader = Data.DataLoader(
    dataset=train_data,
    batch_size=Batch_size,
    shuffle=True,
)

val_loader = Data.DataLoader(
    dataset=val_data,
    batch_size=Batch_size,
    shuffle=False
)


# create net
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_layer):
        super(Net, self).__init__()
        self.input = nn.Linear(n_feature, n_hidden)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(n_hidden, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(n_hidden, n_output)
        self.layernum = n_layer

    def forward(self, x):
        out = self.input(x)
        out = self.relu(out)
        for i in range(self.layernum):
            out = self.hidden(out)
            # out = self.dropout(out)
            out = self.relu(out)
        out = self.out(out)
        return out


Loss_Function = nn.MSELoss()

seed_torch(9)  # 7 9
net = Net(7, hidden_feature, 1, LayerNumber)
net = net.to(device)
optimzer = torch.optim.Adam(
    net.parameters(),
    lr=learning_rate,
)


stepsize = [1800, 1900]  # 1360
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=optimzer,
    milestones=stepsize,
    gamma=0.1
)
# save in excel
workbook = xlsxwriter.Workbook(
    'train_result_error_L%dN%d_Seed=10.xlsx'  # seed=9
    % (LayerNumber, hidden_feature)
)
worksheet = workbook.add_worksheet()
# worksheet2 = workbook.add_worksheet()


worksheet.write('A1', 'epoch')
worksheet.write('B1', 'training loss')
worksheet.write('C1', 'validation loss')
worksheet.write('D1', 'Y_test Data')
worksheet.write('E1', 'Result_test Data')
worksheet.write('F1', 'Relative error')
worksheet.write('G1', 'Average Relative error')
worksheet.write('H1', 'Average Relative error')
worksheet.write('I1', 'Test Data Wn')
worksheet.write('J1', 'Test Data Wp')
worksheet.write('K1', 'Test Data h')
worksheet.write('L1', 'Test Data h_ic')
worksheet.write('M1', 'Test Data FF')
worksheet.write('N1', 'Test Data Q_in')
worksheet.write('O1', 'Test Data Rc')


def TrainGA(epoch):
    # seed_torch(sd)
    for i in range(epoch):
        train_loss = 0.0
        # val_loss = 0.0
        temp_loss = 0.0
        temp_val = 0.0

        net.train()
        for num, (batch_x, batch_y) in enumerate(loader):
            optimzer.zero_grad()
            out = net(batch_x.to(device))
            loss = Loss_Function(out, batch_y.to(device))
            loss.backward()
            optimzer.step()
            temp_loss += loss.item()
        scheduler.step()
        train_loss = temp_loss / (train_size / Batch_size)
        net.eval()
        with torch.no_grad():
            for epnum, (val_x, val_y) in enumerate(val_loader):
                val_out = net(val_x.to(device))
                dev_loss = Loss_Function(val_out, val_y.to(device))
                temp_val += dev_loss.cpu().data.numpy()

        val_loss = temp_val / (valid_size / Batch_size)
        print('epoch: %d' % i, 'training loss:', train_loss, '|',
              'validation loss:', val_loss)
        worksheet.write(i + 1, 0, i + 1)
        worksheet.write(i + 1, 1, train_loss)
        worksheet.write(i + 1, 2, val_loss)
    return train_loss


# start training
def SeedTrain(SeedNumber):
    Num = len(SeedNumber)
    # print(Num)
    result = np.zeros(Num)
    for j in range(Num):
        # print(j)
        result[j] = TrainGA(epoch, SeedNumber[j])
    return result


time_start = time.time()
TrainGA(epoch)
time_end = time.time()
time_cost = time_end - time_start
print('Cost Time', time_cost, 's')
# test data
test_out = net(X_test.to(device))
t_out = test_out.cpu().data.numpy()
num = len(t_out)
Loss_out = np.zeros(num)
result = 0
Y_test = testing_y
t_out = recover_y(t_out)
for i in range(num):
    worksheet.write(i + 1, 3, Y_test[i])
    worksheet.write(i + 1, 4, t_out[i])

    # Loss_out[i] = np.power((t_out[i] - Y_test[i]), 2)
    Loss_out[i] = np.abs(t_out[i] - Y_test[i]) / Y_test[i]
    result += Loss_out[i]

    worksheet.write(i + 1, 5, Loss_out[i])
    worksheet.write(i + 1, 8, testing_x[i][0])
    worksheet.write(i + 1, 9, testing_x[i][1])
    worksheet.write(i + 1, 10, testing_x[i][2])
    worksheet.write(i + 1, 11, testing_x[i][3])
    worksheet.write(i + 1, 12, testing_x[i][4])
    worksheet.write(i + 1, 13, testing_x[i][5])
    worksheet.write(i + 1, 14, testing_x[i][6])
temp = np.sum(Loss_out, axis=0)
Average_error = temp / num
worksheet.write(1, 6, Average_error)

workbook.close()

torch.save(
    net.state_dict(), 'TEGNetP4_L%dN%d.pkl'
    % (LayerNumber, hidden_feature)
)

# print('Finish')
