import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import data_preprocess
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
use_cuda=torch.cuda.is_available()


# 将数据划分为训练集和测试集
X_train, X_test, Y_train, Y_test = data_preprocess.tensorFromData()
trainDataSet = data_preprocess.TextDataSet(X_train, Y_train)
testDataSet = data_preprocess.TextDataSet(X_test, Y_test)
trainDataLoader = DataLoader(trainDataSet, batch_size=16, shuffle=True)
testDataLoader = DataLoader(testDataSet, batch_size=16, shuffle=False)

# 获取字典
word_to_inx, inx_to_word = data_preprocess.get_dic()
len_dic = len(word_to_inx)

# 定义超参数
MAXLEN = 64
input_dim = MAXLEN
emb_dim = 128
num_epoches = 20
batch_size = 16

#定义模型
class TextInception_model(nn.Module):
    def __init__(self,len_dic,emb_dim,input_dim):
        super(TextInception_model,self).__init__()
        self.embed=nn.Embedding(len_dic,emb_dim)#b,64,128
        self.block1=nn.Conv1d(input_dim,128,1,1,0)#b,128,128
        self.block2=nn.Sequential(
            nn.Conv1d(input_dim,256,1,1,padding=0),#b,256,128
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 128, 3, 1, padding=1),  # b,128,128
        )
        self.block3=nn.Sequential(
            nn.Conv1d(input_dim,256,3,1,padding=1),#b,256,128
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 128, 5, 1, padding=2),  # b,128,128
        )
        self.block4=nn.Conv1d(input_dim,128,3,1,1)#b,128,128
        #b,128,128*4
        #b,128*128*4
        self.linear=nn.Linear(128*128*4,128)#b,128
        self.relu=nn.ReLU(True)
        self.drop=nn.Dropout(0.3)#b,128
        self.classify=nn.Linear(128,3)#b,3
    def forward(self, x):
        x=self.embed(x)
        x1=self.block1(x)
        x2=self.block2(x)
        x3=self.block3(x)
        x4=self.block4(x)
        x=torch.cat((x1,x2,x3,x4),2)
        b,c,d=x.size()
        x=x.view(-1,c*d)
        x=self.linear(x)
        x=self.relu(x)
        x=self.drop(x)
        out=self.classify(x)
        return out


if use_cuda:
    model=TextInception_model(len_dic,emb_dim,input_dim).cuda()
else:
    model=TextInception_model(len_dic,emb_dim,input_dim)

criterion = nn.CrossEntropyLoss()
optimzier = optim.Adam(model.parameters(), lr=1e-3)
best_acc=0
best_model=None
for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()
    for i, data in enumerate(trainDataLoader):
        x, y = data
        if use_cuda:
            x, y = Variable(x).cuda(), Variable(y).cuda()
        else:
            x, y = Variable(x), Variable(y)
        # forward
        out = model(x)
        loss = criterion(out, y)
        train_loss += loss.data[0] * len(y)
        _, pre = torch.max(out, 1)
        num_acc = (pre == y).sum()
        train_acc += num_acc.data[0]
        # backward
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        if (i + 1) % 100 == 0:
            print('[{}/{}],train loss is:{:.6f},train acc is:{:.6f}'.format(i, len(trainDataLoader),
                                                                            train_loss / (i * batch_size),
                                                                            train_acc / (i * batch_size)))
    print(
        'epoch:[{}],train loss is:{:.6f},train acc is:{:.6f}'.format(epoch,
                                                                     train_loss / (len(trainDataLoader) * batch_size),
                                                                     train_acc / (len(trainDataLoader) * batch_size)))
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for i, data in enumerate(testDataLoader):
        x, y = data
        if use_cuda:
            x = Variable(x, volatile=True).cuda()
            y = Variable(y, volatile=True).cuda()
        else:
            x = Variable(x, volatile=True)
            y = Variable(y, volatile=True)
        out = model(x)
        loss = criterion(out, y)
        eval_loss += loss.data[0] * len(y)
        _, pre = torch.max(out, 1)
        num_acc = (pre == y).sum()
        eval_acc += num_acc.data[0]
    print('test loss is:{:.6f},test acc is:{:.6f}'.format(
        eval_loss / (len(testDataLoader) * batch_size),
        eval_acc / (len(testDataLoader) * batch_size)))
    if best_acc<(eval_acc / (len(testDataLoader) * batch_size)):
        best_acc=eval_acc / (len(testDataLoader) * batch_size)
        best_model=model.state_dict()
        # print(best_model)
        print('best acc is {:.6f},best model is changed'.format(best_acc))

torch.save(model.state_dict(),'./model/TextInception_model.pth')