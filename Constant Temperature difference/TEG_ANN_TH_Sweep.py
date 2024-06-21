# TEG Constant TH experiment
# ANN data sweep
# Available on https://github.com/LorewalkerZYX/Bulk-TEG-project.git

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
import xlsxwriter

M = -5.713109082133402
S = 1.461078971097606

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


# normalization
def normalize_x(x, input=True):
    temp = x

    if input:
        wn = 4.5  # wn = [0.5-5]
        wp = 4.5  # wp = [0.5-5]
        h = 4.5  # h = [0.5-5]
        h_ic = 2.5  # h_ic = [0.5-3]
        ff = 0.9  # ff = [0.05-0.95]
        t_h = 200  # T_H = [300-500]
        rho_c = 9.9E-8  # rho_c = [1E-9-1E-7]

        for i in range(len(temp)):
            temp[i, 0] = (temp[i, 0] - 0.5) / wn
            temp[i, 1] = (temp[i, 1] - 0.5) / wp
            temp[i, 2] = (temp[i, 2] - 0.5) / h
            temp[i, 3] = (temp[i, 3] - 0.5) / h_ic
            temp[i, 4] = (temp[i, 4] - 0.05) / ff
            temp[i, 5] = (temp[i, 5] - 300) / t_h
            temp[i, 6] = (temp[i, 6] - 1E-9) / rho_c

    else:
        for k in range(len(temp)):
            temp[k][0] = np.log(temp[k][0])
            temp[k][0] = (temp[k][0] - Mp) / Sp
            temp[k][1] = np.log(temp[k][1])
            temp[k][1] = (temp[k][1] - Me) / Se
    return temp


def normalize(x, input=True):
    temp = x
    wn = 4.5  # wn = [0.5-5]
    wp = 4.5  # wp = [0.5-5]
    h = 4.5  # h = [0.5-5]
    h_ic = 2.5  # h_ic = [0.5-3]
    ff = 0.9  # ff = [0.05-0.95]
    t_h = 200  # T_H = [300-500]
    rho_c = 9.9E-8  # rho_c = [1E-9-1E-7]
    temp[0] = (temp[0] - 0.5) / wn
    temp[1] = (temp[1] - 0.5) / wp
    temp[2] = (temp[2] - 0.5) / h
    temp[3] = (temp[3] - 0.5) / h_ic
    temp[4] = (temp[4] - 0.05) / ff
    temp[5] = (temp[5] - 300) / t_h
    temp[6] = (temp[6] - 1E-9) / rho_c
    return temp


def recover_y(y):
    y[0] = y[0] * Sp + Mp
    y[1] = y[1] * Se + Me
    outy = np.exp(y)
    return outy[0]


Batch_size = 64
hidden_feature = 400
Mp = -2.543717849328878  # -3.236864469716249
Sp = 1.956137781915298
Mq = 8.401629926494419
Sq = 1.050063542837697
Me = -4.730739701517070
Se = 1.175501917388629

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
            out = self.relu(out)
        out = self.out(out)
        return out


TEG_NET = Net(7, hidden_feature, 2, 5).to(device)
TEG_NET.load_state_dict(torch.load('TEGNetP4_V2.pkl'))

Qin = range(1000, 5000)

wn = 2.5
wp = 2.5
h = 2.5  # 4.97076821
h_ic = 1.5  # 0.50024453
ff = 0.5
Rate = 10
R0 = 1E-9
rho_c = Rate * R0
T_H = 400

H = np.arange(0.5, 5.01, 0.01)

# save in excel
workbook = xlsxwriter.Workbook('./PowerDensity/Power_H_FF=%.1f.xlsx' % ff)
worksheet = workbook.add_worksheet()
worksheet.write('A1', 'Wn')
worksheet.write('B1', 'Wp')
worksheet.write('C1', 'H_TE')
worksheet.write('D1', 'H_ic')
worksheet.write('E1', 'FF')
worksheet.write('F1', 'Qin')
worksheet.write('G1', 'rho_c')
worksheet.write('H1', 'Power Max')
worksheet.write(1, 0, wn)
worksheet.write(1, 1, wp)
worksheet.write(1, 4, ff)
worksheet.write(1, 3, h_ic)
# worksheet.write(1, 4, ff)
worksheet.write(1, 5, T_H)
worksheet.write(1, 6, rho_c)
# print(len(Qin))
for i in range(len(H)):
    worksheet.write(i + 1, 2, H[i])
    input_x = [wn, wp, H[i], h_ic, ff, T_H, rho_c]
    normalize(input_x)
    # trainsfer numpy to torch
    x = torch.tensor(input_x)
    x = x.type(torch.FloatTensor)
    y = TEG_NET(x.to(device))
    result = recover_y(y.cpu().data.numpy())
    worksheet.write(i + 1, 7, result)
    # print(y.cpu().data.numpy())
workbook.close()
'''
input_x = [wn, wp, h, h_ic, ff, T_H, rho_c]
normalize(input_x)
x = torch.tensor(input_x)
x = x.type(torch.FloatTensor)
y = TEG_NET(x.to(device))
result = recover_y(y.cpu().data.numpy())
print(result)'''
