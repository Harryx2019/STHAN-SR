import os
from time import time
from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn
import torch.optim as optim
from torch_geometric import nn
from scipy import sparse
from torch_geometric import utils
import random

from hgat_nasdaq import HGAT 
from load_data_nasdaq import load_EOD_data
import math
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import ndcg_score
from empyrical.stats import max_drawdown, downside_risk, calmar_ratio

def get_batch(steps, seq_len, eod_data, mask_data, gt_data, price_data ,offset):
    # if offset is None:
    #     offset = random.randrange(0, valid_index)
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return eod_data[:, offset:offset + seq_len, :],\
            np.expand_dims(mask_batch, axis=1),\
            np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),\
            np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1)

def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    # print('gt_rt',np.max(ground_truth))
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2 / np.sum(mask)
    
    bt_long5 = 1.0
    bt_long5_gt = 1.0
    ndcg_score_top5 = 0.0
    sharpe_li5 = []
    irr = []
    selected_stock5 = []

    for i in range(prediction.shape[1]):
        # 返回索引
        rank_gt = np.argsort(ground_truth[:, i])
        # 真实前5名排序
        gt_top5 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)

        # 预测前5名排序
        rank_pre = np.argsort(prediction[:, i])
        pre_top5 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
        # 保存选股
        selected_stock5.append(pre_top5)

        ndcg_score_top5 += ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1))

        # back testing on top 5
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        # 累计收益率计算公式
        bt_long5 *= (1+real_ret_rat_top5)
        sharpe_li5.append(real_ret_rat_top5)
        irr.append(bt_long5)

    performance['btl5'] = bt_long5 - 1
    performance['ndcg_score_top5'] = ndcg_score_top5/prediction.shape[1]
    sharpe_li5 = np.array(sharpe_li5)
    performance['sharpe5'] = (np.mean(sharpe_li5)/np.std(sharpe_li5))*15.87 #To annualize

    # 返回技术指标[mse,ndcg_score_top5,btl5(累计收益率),sharpe5(夏普比率)]
    # irr: 收益率序列
    # selected_stock5: 选股序列
    return performance,irr,selected_stock5


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def trr_loss_mse_rank(pred, base_price, ground_truth, mask, alpha, no_stocks,device):
    return_ratio = torch.div((pred- base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    all_ones = torch.ones(no_stocks,1).to(device)
    pre_pw_dif =  (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1)) 
                    - torch.matmul(all_ones, torch.transpose(return_ratio, 0, 1)))
    gt_pw_dif = (
            torch.matmul(all_ones, torch.transpose(ground_truth,0,1)) -
            torch.matmul(ground_truth, torch.transpose(all_ones, 0,1))
        )

    mask_pw = torch.matmul(mask, torch.transpose(mask, 0,1))
    rank_loss = torch.mean(
            F.relu(
                ((pre_pw_dif*gt_pw_dif)*mask_pw)))
    loss = reg_loss + alpha*rank_loss
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, reg_loss, rank_loss, return_ratio

def train(data_path,market_name,tickers_fname,rel_data_path,parameters):
    # load data
    tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                            dtype=str, delimiter='\t', skip_header=False)
    print('#tickers selected:', len(tickers))

    # 加载股票时序数据
    steps = parameters['steps']
    seq_len = parameters['seq']
    eod_data, mask_data, gt_data, price_data = load_EOD_data(data_path, market_name, tickers, steps)
    print('#tickers feature:', eod_data.shape)

    # 加载股票关系数据
    inci_mat = np.load(rel_data_path)
    print("#inci_mat:",inci_mat.shape)
    inci_sparse = sparse.coo_matrix(inci_mat)
    print("#inci_sparse:",inci_sparse.shape)
    # 每条边的权重都为1
    # 2*edge_num
    incidence_edge = utils.from_scipy_sparse_matrix(inci_sparse)
    print("#incidence_edge:",incidence_edge[0].shape)

    # 定义设备
    gpu = parameters['gpu']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #单卡gpu
    if gpu == True:
        device_name = torch.cuda.get_device_name()
    else:
        device_name = '/cpu:0'
    print('#device name:', device_name)

    # 数据集划分
    valid_index = 756
    test_index = 1008
    trade_dates = mask_data.shape[1]
    fea_dim = 5
    num_edges = inci_mat.shape[1]
    batch_size = len(tickers) 
    
    # 图结构输入
    hyp_input = incidence_edge[0].to(device)

    tra_loss_grid = [] # 记录每个grid的训练损失
    tra_reg_loss_grid = [] # 记录每个训练grid的reg_loss
    tra_rank_loss_grid = [] # 记录每个训练grid的rank_loss

    val_loss_grid = [] # 记录每个grid的验证损失
    val_reg_loss_grid = [] # 记录每个grid的验证reg_loss
    val_rank_loss_grid = [] # 记录每个grid的验证rank_loss

    val_output_grid = [] # 记录每个grid预测收盘价
    val_pred_grid = [] # 记录每个grid预测收益率序列

    val_ndcg_score_top5_grid = [] #记录每个grid的验证ndcg5
    val_btl5_grid = [] # 记录每个grid的验证集累计收益率
    val_irr_grid = [] # 记录每个grid的验证累计收益率序列
    val_sharpe5_grid = [] # 记录每个grid的验证集夏普比率

    # 批量读取数据
    batch_offsets = np.arange(start=0, stop=valid_index - parameters['seq'] - steps + 1 , dtype=int)

    # 网格搜索超参数
    # for lr in [0.0001,0.0005,0.0007,0.0009,0.001,0.003,0.005]:
    #     for alpha in [1,2,3,4,5,6,7,8,9,10]:
    for lr in [0.005]:
        for alpha in [3]:
            parameters['lr'] = lr
            parameters['alpha'] = alpha
            
            print('#lr={},alpha={}'.format(lr,alpha))
            
            # 定义模型
            model = HGAT(batch_size).to(device)
            # 初始化参数，设置优化器
            for p in model.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    torch.nn.init.uniform_(p)
            optimizer_hgat = optim.Adam(model.parameters(), lr=parameters['lr'], weight_decay=5e-4)
            
            tra_loss_epoch = [] # 记录每个epoch的训练损失
            tra_reg_loss_epoch = [] # 记录每个训练epoch的reg_loss
            tra_rank_loss_epoch = [] # 记录每个训练epoch的rank_loss

            val_loss_epoch = [] # 记录每个epoch的损失
            val_reg_loss_epoch = [] # 记录每个训练epoch的验证reg_loss
            val_rank_loss_epoch = [] # 记录每个训练epoch的验证rank_loss

            val_pred_epoch = [] # 记录每个epoch预测收益率序列
            val_output_epoch = [] # 记录每个epoch预测收盘价序列

            val_ndcg_score_top5_epoch = [] #记录每个训练epoch的验证ndcg5
            val_btl5_epoch = [] # 记录每个训练epoch的验证集累计收益率
            val_irr_epoch = [] # 记录每个训练epoch的验证集预测top5收益率序列
            val_sharpe5_epoch = [] # 记录每个epoch的验证集夏普比率
            
            epochs = parameters['epochs']
            # 每个epoch训练
            for i in range(epochs):
                # 打乱索引下标
                np.random.shuffle(batch_offsets)
                tra_loss = 0.0
                tra_reg_loss = 0.0
                tra_rank_loss = 0.0
                model.train() 
                for j in tqdm(range(valid_index - parameters['seq'] - steps +1)):
                    # 获取一个批次的数据
                    emb_batch, mask_batch, price_batch, gt_batch = get_batch(steps,seq_len,eod_data, mask_data, gt_data, price_data, batch_offsets[j])

                    optimizer_hgat.zero_grad()
                    output = model(torch.FloatTensor(emb_batch).to(device), hyp_input, num_edges, device)
                    cur_loss, cur_reg_loss, cur_rank_loss, curr_rr_train = trr_loss_mse_rank(output.reshape((1026,1)), torch.FloatTensor(price_batch).to(device), 
                                                                                            torch.FloatTensor(gt_batch).to(device), 
                                                                                            torch.FloatTensor(mask_batch).to(device), 
                                                                                            parameters['alpha'], batch_size,device)
                    tra_loss += cur_loss.item()
                    tra_reg_loss += cur_reg_loss.item()
                    tra_rank_loss += cur_rank_loss.item()
                    cur_loss.backward()
                    optimizer_hgat.step()

                print('Train Loss:',
                    tra_loss / (valid_index - parameters['seq'] - steps + 1),
                    tra_reg_loss / (valid_index - parameters['seq'] - steps + 1),
                    tra_rank_loss / (valid_index - parameters['seq'] - steps + 1))
                # 记录
                tra_loss_epoch.append(tra_loss / (valid_index - parameters['seq'] - steps + 1))
                tra_reg_loss_epoch.append(tra_reg_loss / (valid_index - parameters['seq'] - steps + 1))
                tra_rank_loss_epoch.append(tra_rank_loss / (valid_index - parameters['seq'] - steps + 1))

                with torch.no_grad():
                    # test on validation set
                    cur_valid_output = np.zeros([len(tickers), test_index - valid_index],dtype=float)
                    cur_valid_pred = np.zeros([len(tickers), test_index - valid_index],dtype=float)
                    cur_valid_gt = np.zeros([len(tickers), test_index - valid_index],dtype=float)
                    cur_valid_mask = np.zeros([len(tickers), test_index - valid_index],dtype=float)
                    val_loss = 0.0
                    val_reg_loss = 0.0
                    val_rank_loss = 0.0
                    model.eval()
                    for cur_offset in range(valid_index - parameters['seq'] - steps + 1,
                                            test_index - parameters['seq'] - steps + 1):
                        emb_batch, mask_batch, price_batch, gt_batch = get_batch(steps,seq_len,eod_data, mask_data, gt_data, price_data, cur_offset)

                        output_val = model(torch.FloatTensor(emb_batch).to(device), hyp_input, num_edges,device)
                        cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(output_val, torch.FloatTensor(price_batch).to(device), 
                                                                                                torch.FloatTensor(gt_batch).to(device), 
                                                                                                torch.FloatTensor(mask_batch).to(device), 
                                                                                                parameters['alpha'], batch_size,device)

                        cur_rr = cur_rr.detach().cpu().numpy().reshape((1026,1))
                        output_val = output_val.detach().cpu().numpy().reshape((1026,1))

                        val_loss += cur_loss.detach().cpu().item()
                        val_reg_loss += cur_reg_loss.detach().cpu().item()
                        val_rank_loss += cur_rank_loss.detach().cpu().item()

                        cur_valid_output[:, cur_offset - (valid_index - parameters['seq'] - steps + 1)] = copy.copy(output_val[:, 0])
                        cur_valid_pred[:, cur_offset - (valid_index - parameters['seq'] - steps + 1)] = copy.copy(cur_rr[:, 0])
                        cur_valid_gt[:, cur_offset - (valid_index - parameters['seq'] - steps + 1)] = copy.copy(gt_batch[:, 0])
                        cur_valid_mask[:, cur_offset - (valid_index - parameters['seq'] - steps + 1)] = copy.copy(mask_batch[:, 0])
                    print('Valid MSE:',
                        val_loss / (test_index - valid_index),
                        val_reg_loss / (test_index - valid_index),
                        val_rank_loss / (test_index - valid_index))
                    cur_valid_perf,irr,selected_stock5 = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
                    print('Valid preformance:', cur_valid_perf)
                    # 记录
                    val_loss_epoch.append(val_loss / (test_index - valid_index)) 
                    val_reg_loss_epoch.append(val_reg_loss / (test_index - valid_index)) 
                    val_rank_loss_epoch.append(val_rank_loss / (test_index - valid_index)) 

                    val_pred_epoch.append(cur_valid_pred)
                    val_output_epoch.append(cur_valid_output)
                    val_ndcg_score_top5_epoch.append(cur_valid_perf['ndcg_score_top5']) 
                    val_btl5_epoch.append(cur_valid_perf['btl5'])
                    val_irr_epoch.append(irr)
                    val_sharpe5_epoch.append(cur_valid_perf['sharpe5']) 

                    # test on testing set
                    # cur_test_output = np.zeros([len(tickers), trade_dates - test_index],dtype=float)
                    # cur_test_pred = np.zeros([len(tickers), trade_dates - test_index],dtype=float)
                    # cur_test_gt = np.zeros([len(tickers), trade_dates - test_index],dtype=float)
                    # cur_test_mask = np.zeros([len(tickers), trade_dates - test_index],dtype=float)
                    # test_loss = 0.0
                    # test_reg_loss = 0.0
                    # test_rank_loss = 0.0
                    # model.eval()
                    # for cur_offset in range(test_index - parameters['seq'] - steps + 1,
                    #                         trade_dates - parameters['seq'] - steps + 1):
                    #     emb_batch, mask_batch, price_batch, gt_batch = get_batch(steps,seq_len,eod_data, mask_data, gt_data, price_data, cur_offset)

                    #     output_test = model(torch.FloatTensor(emb_batch).to(device), hyp_input, num_edges,device)
                    #     cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(output_test, torch.FloatTensor(price_batch).to(device), 
                    #                                                                             torch.FloatTensor(gt_batch).to(device), 
                    #                                                                             torch.FloatTensor(mask_batch).to(device), 
                    #                                                                             parameters['alpha'], batch_size,device)

                    #     output_test = output_test.detach().cpu().numpy().reshape((1026,1))
                    #     cur_rr = cur_rr.detach().cpu().numpy().reshape((1026,1))

                    #     test_loss += cur_loss.detach().cpu().item()
                    #     test_reg_loss += cur_reg_loss.detach().cpu().item()
                    #     test_rank_loss += cur_rank_loss.detach().cpu().item()

                    #     cur_test_output[:, cur_offset - (test_index - parameters['seq'] - steps + 1)] = copy.copy(output_test[:, 0])
                    #     cur_test_pred[:, cur_offset - (test_index - parameters['seq'] - steps + 1)] = copy.copy(cur_rr[:, 0])
                    #     cur_test_gt[:, cur_offset - (test_index - parameters['seq'] - steps + 1)] = copy.copy(gt_batch[:, 0])
                    #     cur_test_mask[:, cur_offset - (test_index - parameters['seq'] - steps + 1)] = copy.copy(mask_batch[:, 0])
                    # print('Test MSE:',
                    #     test_loss / (trade_dates - test_index),
                    #     test_reg_loss / (trade_dates - test_index),
                    #     test_rank_loss / (trade_dates - test_index))
                    # cur_test_perf,irr,selected_stock5 = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
                    # print('Test performance:', cur_test_perf)

                    # # 记录每轮epoch结果
                    # val_loss_epoch.append(test_loss / (trade_dates - test_index)) 
                    # val_reg_loss_epoch.append(test_reg_loss / (trade_dates - test_index)) 
                    # val_rank_loss_epoch.append(test_rank_loss / (trade_dates - test_index))
                    # # 预测收益率
                    # val_pred_epoch.append(cur_test_pred)
                    # val_output_epoch.append(cur_test_output)
                    # # 排序指标
                    # val_ndcg_score_top5_epoch.append(cur_test_perf['ndcg_score_top5']) 
                    # # 累计收益率
                    # val_irr_epoch.append(irr)
                    # # 累计收益率序列
                    # val_btl5_epoch.append(cur_test_perf['btl5'])
                    # # 夏普比率
                    # val_sharpe5_epoch.append(cur_test_perf['sharpe5']) 

            # 训练损失
            tra_loss_grid.append(tra_loss_epoch)
            tra_reg_loss_grid.append(tra_reg_loss_epoch)
            tra_rank_loss_grid.append(tra_rank_loss_epoch)
            # 验证损失
            val_loss_grid.append(val_loss_epoch)
            val_reg_loss_grid.append(val_reg_loss_epoch)
            val_rank_loss_grid.append(val_rank_loss_epoch)
            # 收益率序列
            val_pred_grid.append(val_pred_epoch)
            val_output_grid.append(val_output_epoch)
            # 排名指标
            val_ndcg_score_top5_grid.append(val_ndcg_score_top5_epoch)
            # 累计收益率
            val_btl5_grid.append(val_btl5_epoch)
            # 累计收益率序列
            val_irr_grid.append(val_irr_epoch)
            # 夏普比率
            val_sharpe5_grid.append(val_sharpe5_epoch)

    # 保存结果
    res_path = '../result4'
    np.save(os.path.join(res_path,'tra_loss_grid'),tra_loss_grid)
    np.save(os.path.join(res_path,'val_loss_grid'),val_loss_grid)

    np.save(os.path.join(res_path,'val_pred_grid'),val_pred_grid)
    np.save(os.path.join(res_path,'val_output_grid'),val_output_grid)

    np.save(os.path.join(res_path,'val_ndcg_score_top5_grid'),val_ndcg_score_top5_grid)
    np.save(os.path.join(res_path,'val_btl5_grid'),val_btl5_grid)
    np.save(os.path.join(res_path,'val_irr_grid'),val_irr_grid)
    np.save(os.path.join(res_path,'val_sharpe5_grid'),val_sharpe5_grid)




if __name__ == '__main__':
    # 数据目录
    data_path = '../data/2013-01-01'
    market_name = 'NASDAQ'
    tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'

    rel_data_path = '../data/relation/NASDAQ_relation.npy'

    # 定义参数
    parameters = {'seq': 16, # length of historical sequence for feature
                'unit': 64, # number of hidden units in lstm
                'lr': 0.001, # learning rate
                'alpha': 1, # the weight of ranking loss
                'steps':1, # 单步预测
                'epochs':500,
                'gpu':True,
                }
    
    train(data_path,market_name,tickers_fname,rel_data_path,parameters)

