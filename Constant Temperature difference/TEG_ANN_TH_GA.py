# TEG Constant TH GA experiment
# Available on https://github.com/LorewalkerZYX/Bulk-TEG-project.git

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sko.GA import GA
import xlsxwriter

Batch_size = 64
epoch = 2000
learning_rate = 0.001
hidden_feature = 400
Mp = -2.543717849328878
Sp = 1.956137781915298
Mq = 8.401629926494419
Sq = 1.050063542837697
Me = -4.730739701517070
Se = 1.175501917388629


# Set the random seed manually for reproducibility.
def seed_torch(seed=1029):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # random.seed(seed)
    np.random.seed(seed)


seed_torch(0)  # 30,


def recover_y(y):
    y[0] = y[0] * Sp + Mp
    y[1] = y[1] * Se + Me
    outy = np.exp(y)
    return outy


def selection_tournament(self, tourn_size=4):
    '''
    Select the best individual among *tournsize* randomly chosen
    individuals,
    :param self:
    :param tourn_size:
    :return:
    '''
    FitV = self.FitV
    sel_index = []
    for i in range(self.size_pop):
        aspirants_index = np.random.choice(range(self.size_pop), size=tourn_size)
        # aspirants_index = np.random.randint(self.size_pop, size=tourn_size)
        sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
    self.Chrom = self.Chrom[sel_index, :]  # next generation
    return self.Chrom


def ranking(self):
    # GA select the biggest one, but we want to minimize func, so we put a negative here
    self.FitV = (self.Y - np.argmin(self.Y))  # self.Y  # [np.argsort(1 - self.Y)]
    return self.FitV


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


TEG_NET = Net(7, hidden_feature, 2, 5)
TEG_NET.load_state_dict(torch.load('TEGNetP4_V2.pkl'))


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


def normalize_new(x, input=True):
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


# denormalization
def denormalize_x(x):
    temp = x.copy()
    wn = 4.5  # wn = [0.5-5]
    wp = 4.5  # wp = [0.5-5]
    h = 4.5  # h = [0.5-5]
    h_ic = 2.5  # h_ic = [0.5-3]
    ff = 0.9  # ff = [0.05-0.95]
    # q_in = 4000  # q_in = [1000-5000]
    # rho_c = 9.9E-8  # rho_c = [1E-9-1E-7]

    temp[0] = temp[0] * wn + 0.5
    temp[1] = temp[1] * wp + 0.5
    temp[2] = temp[2] * h + 0.5
    temp[3] = temp[3] * h_ic + 0.5
    temp[4] = temp[4] * ff + 0.05
    # temp[5] = temp[5] * q_in + 1000
    # temp[6] = temp[6] * rho_c + 1E-9

    return temp


Th = 0.5  # Th = 400
R_c = 1/11  # Rho_c = 1E-8

T_H = 400
Rhoc = 1E-8


def demo_func(x):
    # print(x[0, :])
    # x.reshape(4, 300)
    # temp = normalize_x(x)
    y = x / 100
    In = np.append(y, T_H)
    InputX = np.append(In, Rhoc)
    InputX = normalize_new(InputX)
    temp = torch.Tensor(InputX)
    # x1, x2, x3, x4 = temp
    # InX = normalize_new(temp)
    # print(InX)
    result = TEG_NET(temp)
    tempy = result.cpu().data.numpy()
    outy = recover_y(tempy)[1]
    return outy


all_history_X = []


def maxrun(self, max_iter=None):
    self.max_iter = max_iter or self.max_iter
    for i in range(self.max_iter):
        self.X = self.chrom2x(self.Chrom)
        self.Y = self.x2y()
        self.ranking()
        self.selection()
        self.crossover()
        self.mutation()

        # record the best ones
        generation_best_index = self.FitV.argmax()
        self.generation_best_X.append(self.X[generation_best_index, :])
        self.generation_best_Y.append(self.Y[generation_best_index])
        self.all_history_Y.append(self.Y)
        self.all_history_FitV.append(self.FitV)
        all_history_X.append(self.X)

    global_best_index = np.array(self.generation_best_Y).argmax()
    self.best_x = self.generation_best_X[global_best_index]
    self.best_y = self.func(np.array([self.best_x]))
    return self.best_x, self.best_y


leastB = [0, 0, 0, 0, 0]
MostB = [1, 1, 1, 1, 1]

leastB1 = [50, 50, 50, 50, 5]
MostB1 = [500, 500, 500, 300, 95]
# wn, wp, H, hic, ff
#
ga = GA(
    func=demo_func,
    n_dim=5, size_pop=100,
    max_iter=200,
    lb=leastB1,
    ub=MostB1,
    precision=1
)
# ga.register(operator_name='selection', operator=selection_tournament)
ga.register(operator_name='ranking', operator=ranking)

GA.run = maxrun


best_x, best_y = ga.run()
# origin_x = denormalize_x(best_x)
# print(origin_x)
print(best_x)
print(best_x/100)
print(best_y)

Y_history = pd.DataFrame(ga.all_history_Y)
# X_history = pd.DataFrame(all_history_X[199])
'''
fig, ax = plt.subplots(2, 1)
ax[0].plot(X_history.index, X_history.values[:, 0], '.', color='red')
X_history.max(axis=1).cummax().plot(kind='line')
plt.show()
'''

# print(History_values[:, 0])

# save in excel
workbook = xlsxwriter.Workbook('MaxGA_200G_Eff_All_X.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write('A1', 'Numbers')
worksheet.write('B1', 'Wn')
worksheet.write('C1', 'Wp')
worksheet.write('D1', 'H')
worksheet.write('E1', 'Hic')
worksheet.write('F1', 'FF')
# length = len(History_values)
for i in range(200):
    X_history = pd.DataFrame(all_history_X[i])
    History_index = X_history.index
    History_values = X_history.values
    # worksheet.write(i+1, 1, np.max(History_values[i]))
    for j in range(100):
        worksheet.write(100*i+j+1, 0, 100*i+j+1)
        worksheet.write(100*i+j+1, 1, History_values[j, 0]/100)
        worksheet.write(100*i+j+1, 2, History_values[j, 1]/100)
        worksheet.write(100*i+j+1, 3, History_values[j, 2]/100)
        worksheet.write(100*i+j+1, 4, History_values[j, 3]/100)
        worksheet.write(100*i+j+1, 5, History_values[j, 4]/100)
workbook.close()
