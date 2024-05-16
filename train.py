"""
@FileName：train.py
@Description：sage训练过程
@Author：HaoBoli
"""
import os

from utils.parser import *
from utils.data_utils import *
from utils.utils import *
from model import *
from LightGCN import *
from tqdm import tqdm
from torch.autograd import Variable

import datetime
from collections import defaultdict
#---------------------------#
# 准备参数及日志文件
#---------------------------#
# 获取当前日期和时间
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
args = parse_args()
# 将 Namespace 对象转换为字典
args_dict = vars(args)
# 将参数内容转换为字符串
params_string = ', '.join([f'{key}: {value}' for key, value in args_dict.items()])
print(args)
if not os.path.exists('./log'):
    os.makedirs('./log')
if not os.path.exists('./weight'):
    os.makedirs('./weight')
with open('./log/{}log.txt'.format(formatted_datetime), 'a') as file:
    file.write(params_string + '\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#--------------------------#
# 准备数据：图，训练数据以及测试数据
#--------------------------#
data, data_args, train_data_loader, test_data_loader = get_data(device,args)

#--------------------------------#
# 初始化模型模型以及优化器
#--------------------------------#
if args.model == 'light':
    model = Model_lgcn(args=args, data_args=data_args,device=device)
else:
    model = Model(args=args, data_args=data_args,device=device)
model = model.to(device)
criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#-------------------------------#
# 开始训练过程
#-------------------------------#
best_p10 = 0
best_pth = None
best_epoch = 0
evaluation = ['P5', 'P10', 'P20', 'R5', 'R10', 'R20', 'N5', 'N10', 'N20']
for epoch in range(args.epochs):
    model.train()
    loss_ = 0
    train_data_loader = tqdm(train_data_loader)
    num_train_batch = len(train_data_loader)
    for i, (samples, syn, labels) in enumerate(train_data_loader):
        samplesV = Variable(samples.to(device))
        synV = Variable(syn.to(device))
        labelsV = Variable(labels.to(device))

        optimizer.zero_grad()
        out = model(
            data.x_dict,
            data.edge_index_dict,
            samplesV, synV)
        loss_bce = criterion(out,labelsV)
        if args.numSym > 1:
            reg_loss = model.sym_regular()
        else:
            reg_loss = 0

        loss = loss_bce + args.reg * reg_loss
        loss.backward()
        optimizer.step()
        loss_ += loss

    print('loss=', loss_.detach().cpu().numpy()/num_train_batch)

    local_history = dict()
    temp_history = defaultdict(list)
    if (epoch + 1) % args.test_num ==0:
        model.eval()

        test_data_loader = tqdm(test_data_loader)
        for i, (samples, syn, labels) in enumerate(test_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.to(device))
                synV = Variable(syn.to(device))
                y_gt = labels.squeeze().detach().cpu().numpy()
                scores = model(
                    data.x_dict,
                    data.edge_index_dict,
                    samplesV, synV)
                result = torch.sigmoid(scores).squeeze().detach().cpu().numpy()

                precision_result = multi_label_metric(np.array(y_gt), np.array(result))

                # 记录结果
                for i, pr in enumerate(precision_result):
                    temp_history[evaluation[i]].append(pr)

        for key, value in temp_history.items():
            local_history[key] = float(np.mean(value)) * 100

        history_s = print_local_history(epoch,local_history,args.epochs)

        if local_history['P10'] > best_p10:
            best_p10 = local_history['P10']
            best_pth = model.state_dict()
            best_epoch = epoch + 1
            with open('./log/{}log.txt'.format(formatted_datetime), 'a') as file:
                file.write(history_s + '\n')

torch.save(best_pth, 'weight/'+args.weight_name+str(best_epoch)+'-'+formatted_datetime+'-'+str(round(best_p10, 2))+'.pth')