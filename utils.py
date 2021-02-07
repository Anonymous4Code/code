import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import  precision_score, recall_score, roc_auc_score, accuracy_score, jaccard_score #jaccard_similarity_score, roc_auc_score
from torch.autograd import Variable
import matplotlib.pyplot as plt
import networkx as nx
#from sklearn.decomposition import PCA
import copy
import torch.nn as nn
plt.switch_backend('agg')

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

def traj_padding(sequences, output_dim):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    max_len = max([len(s) for s in sequences])
    out_dims = (num, max_len, output_dim)
    out_tensor = np.zeros(out_dims)
    mask_dim = (num, max_len)
    mask = np.zeros(mask_dim)
    for i, tensor in enumerate(sequences):
        length = len(tensor)
        tensor_arr = np.array(tensor)
        out_tensor[i, :length] = tensor_arr[:, :output_dim]
        mask[i, :length] = 1
    return out_tensor, mask, max_len

def option_padding(sequences):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    max_len = max([len(s) for s in sequences])
    out_dims = (num, max_len, 1)
    out_tensor = np.zeros(out_dims)
    mask_dim = (num, max_len)
    mask = np.zeros(mask_dim)
    for i, tensor in enumerate(sequences):
        length = len(tensor)
        tensor_arr = np.array(tensor).reshape(-1,1)
        out_tensor[i, :length] = tensor_arr
        mask[i, :length] = 1
    return out_tensor, mask

def get_option(state):
    '''return: different room different option'''
    state_0 = state[0]
    state_1 = state[1]
    if state_0 <= 6 and state_1 <= 6:
        op = 0
    if state_0 > 6 and state_1 <= 6:
        op = 1
    if state_0 <= 7 and state_1 > 6:
        op = 2
    if state_0 > 7 and state_1 > 6:
        op = 3
    return op

def get_option_from_seqs(seqs):
    #100 seqs; 1seq->time*traj*8; 1traj-> (s,a) pair 
    #time step of seq-.time stemp to LSTM
    #LSTM input: time stemp embedding*batchsize
    #output: time stemp*1*batchsize
    '''input:'''
    '''return: micro action/lstm'''
    #warm start for LSTM
    op_list_all = []
    for seq in seqs:
        op_list_temp = []
        l_temp = []
        for count,ti in enumerate(seq,start=1):
            #print(ti)
            l_temp.append(ti)
            if ti[0] == 3 and ti[1] == 6:
                op = 0
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 6 and ti[1] == 2:
                op = 1
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 7 and ti[1] == 9:
                op = 2
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 10 and ti[1] == 6:
                op = 3
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif count == len(seq):
                # Use the last confirmed option in this sequence
                op_list_temp += len(l_temp) * [op]
                l_temp = []
        op_list_all += op_list_temp
    return op_list_all

def traj_to_option_list_meta_task(traj):
    '''load traj from traj_file; seg/transfer traj to option idx; save option idx to option_file'''
    option_list = []
    traj_arr = []
    for i in range(traj.shape[0]):
        for j in range(len(traj[i])):
            traj_arr.append(traj[i][j])
         #   option_list.append(get_option(traj[i][j][:2]))
    traj_arr = np.array(traj_arr)
   # option_arr = np.array(option_list).reshape(len(option_list),1)
    return traj_arr#, option_arr

def traj_to_option_list(traj_file_name):
    '''load traj from traj_file; seg/transfer traj to option idx; save option idx to option_file'''
    traj = np.load(traj_file_name, allow_pickle = True)
    option_list = []
    traj_arr = []
    for i in range(traj.shape[0]):
        for j in range(len(traj[i])):
            traj_arr.append(traj[i][j])
            option_list.append(get_option(traj[i][j][:2]))
    traj_arr = np.array(traj_arr)
    option_arr = np.array(option_list).reshape(len(option_list),1)
    return traj_arr, option_arr

def traj_to_option_list_from_seq(traj_file_name):
    '''load traj from traj_file; seg/transfer traj to option idx new way; save option idx to option_file'''
    traj = np.load(traj_file_name, allow_pickle = True)
    option_list = []
    traj_arr = []
    traj_arr = []
    for i in range(traj.shape[0]):
        for j in range(len(traj[i])):
            traj_arr.append(traj[i][j])
    traj_arr = np.array(traj_arr)
    option_list = get_option_from_seqs(traj)
            #option_list.append(get_option(traj[i][j][:2]))
    traj_arr = np.array(traj_arr)
    option_arr = np.array(option_list).reshape(len(option_list),1)
    return traj_arr, option_arr

def traj_to_option_emb(traj_file_name, option_file_name, option_embedding):
    '''load traj from traj_file; seg/transfer traj to option idx; get option emb w.r.t option emb;
    save option embeddin to option_file'''
    traj = np.load(traj_file_name, allow_pickle = True)

    traj_option_emb = []

    for i in range(traj.shape[0]):
        for j in range(len(traj[i])):
            option_emb_given_index = option_embedding[get_option(traj[i][j][:2])]
            traj_option_emb.append(option_emb_given_index)
    traj_option_emb = np.array(traj_option_emb)#.reshape(len(option_list),10)
    np.save(option_file_name, traj_option_emb)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum() # only difference


def evaluation(policy, state, action_expert, option, action_dim):
    actions_pre = []
    actions_evl = []
    action_all_pre = []

    st = torch.tensor(state, dtype=torch.float32)
    option = torch.tensor(option, dtype=torch.float32)

    actions = policy.forward(st, option).data.numpy()
  #  print(actions)
    action_all_pre.append(actions)
    ma_aucs = []
    mi_aucs = []
    for ind in range(actions.shape[0]):
        act = actions[ind]
        a = softmax(act)
        actions_pre.append(a)
        ac_r = action_expert[ind]
        actions_evl.append(ac_r)
    none_zero_index = np.sum(actions_evl, axis=0)
    mask = np.where(none_zero_index>0)
    actions_evl_masked = []
    actions_pre_masked = []


    for ind in range(actions.shape[0]):
        actions_evl_masked.append(actions_evl[ind][mask])
        actions_pre_masked.append(actions_pre[ind][mask])
    
    mi_auc = roc_auc_score(actions_evl_masked, actions_pre_masked, average='micro')
    return mi_auc, action_expert, actions_pre#, ma_auc
   
def random_init_option(option_num, emb_dim):
    option_emb = []
    for i in range(option_num):
        option_emb.append(np.random.uniform(-1, 1, emb_dim))
    #traj_to_option_emb(traj_file_name, option_file_name, option_emb)
    return option_emb

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def get_graph_rec_send(option_num):
    off_diag = np.ones([option_num, option_num]) - np.eye(option_num) #N*N 
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32) #(N)*(N-1)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)
    input_rel_rec = Variable(rel_rec)
    input_rel_send = Variable(rel_send)
    return input_rel_rec, input_rel_send

def convert_to_graph(logits, num_nodes):
    graph = np.zeros([num_nodes, num_nodes])
    for i in range(num_nodes):
        for j in range(num_nodes-1):
            if j >= i:
                graph[i][j+1] = logits[i][j]
            elif j<i:
                graph[i][j] = logits[i][j]
    return graph

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)
    
def graph_to_adj(graph, num_nodes, edge_nums):
    '''Convert top edge_num to 1'''
    edges = np.zeros([num_nodes, num_nodes])
    #largest_indices(graph, edge_nums)
    temp = copy.copy(graph)
    MIN = -1<<10
    for i in range(num_nodes):
    #    edges[i, i] = 1
        temp[i][i] = MIN
    edges[largest_indices(temp, edge_nums)] = 1
    return edges

def visualize_graph_option(vis_folder, cur_implicit_graph, option_emb, epoch):
    option_emb = option_emb.cpu().detach().numpy()
    option_num = len(option_emb)
    c_labels = range(len(option_emb))

    for edge_num in range(3,9):
        edgelist = graph_to_adj(cur_implicit_graph, option_num, edge_num)
        #vis graph
        G = nx.DiGraph(edgelist)
        nx.draw(G, node_size=1000, width=2, with_labels=True, pos=nx.spring_layout(G))
        plt.show(block=False)
        output_res_name = vis_folder+"Epoch_"+str(epoch)+"Edge_"+str(edge_num)+".png"
        plt.savefig(output_res_name, format="PNG")
        plt.close()
    
  
def get_option_from_seqs_lstm(seqs):
    '''input:'''
    '''return: micro action/lstm'''
    #warm start for LSTM
    op_list_all = []
    for seq in seqs:
        op_list_temp = []
        l_temp = []
        for count,ti in enumerate(seq,start=1):
            #print(ti)
            l_temp.append(ti)
            if ti[0] == 3 and ti[1] == 6:
                op = 0
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 6 and ti[1] == 2:
                op = 1
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 7 and ti[1] == 9:
                op = 2
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 10 and ti[1] == 6:
                op = 3
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif count == len(seq):
                # Use the last confirmed option in this sequence
                op_list_temp += len(l_temp) * [op]
                l_temp = []
        op_list_all.append(op_list_temp)
    return op_list_all

def convert_traj_lstm_opt_arr(predicted_opt, input_traj_mask, traj_list):
    #convert the LSTM logits to opt via argmax
    predicted_opt = torch.argmax(predicted_opt, dim=2)
    mask_opt = input_traj_mask.view(-1)
    #np.savetxt("mask_opt.txt", mask_opt, delimiter=',', fmt='%d')
    predicted_opt = predicted_opt.view(-1)
    #np.savetxt("option_arr.txt", predicted_opt, delimiter=',', fmt='%d')
   # print(predicted_opt)
    option_idx = torch.zeros(len(traj_list))
    init_opt_idx = 0
    #print(option_idx)
    for i in range(mask_opt.shape[0]):
        if mask_opt[i] == 1:
            #print(i, predicted_opt[i])
            option_idx[init_opt_idx] = predicted_opt[i]
            init_opt_idx+=1
    option_idx = Variable(option_idx.long())
    return option_idx

def traj_padding_together(sequences_meta, output_dim):
    """
    :param sequences: list of tensors
    :return:
    """
    max_meta_Len = 0
    out_tensors = []
    masks = []
    traj_length = 0
    for sequences in sequences_meta:
        num = len(sequences)
        max_len = max([len(s) for s in sequences])
        traj_length+= sum([len(s) for s in sequences])
        max_meta_Len = max(max_len, max_meta_Len)
    for sequences in sequences_meta:
        out_dims = (num, max_meta_Len, output_dim)
        out_tensor = np.zeros(out_dims)
        mask_dim = (num, max_meta_Len)
        mask = np.zeros(mask_dim)
        for i, tensor in enumerate(sequences):
            length = len(tensor)
            tensor_arr = np.array(tensor)
            out_tensor[i, :length] = tensor_arr[:, :output_dim]
            mask[i, :length] = 1
        out_tensors.append(out_tensor)
        masks.append(mask)
    out_tensor = out_tensors[0]
    mask = masks[0]
    for i in range(1, len(out_tensors)):
        out_tensor = np.concatenate((out_tensor, out_tensors[i]), axis=0)
        mask = np.concatenate((mask, masks[i]), axis=0)

    return out_tensor, mask, max_meta_Len, traj_length


def traj_padding_meta_task(sequences, output_dim):
    """
    :param sequences: list of tensors
    :return:
    """
    max_meta_Len = 0
    out_tensors = []
    masks = []
    traj_length = 0
    num = len(sequences)
    max_len = max([len(s) for s in sequences])
    traj_length+= sum([len(s) for s in sequences])
    max_meta_Len = max(max_len, max_meta_Len)
    out_dims = (num, max_meta_Len, output_dim)
    out_tensor = np.zeros(out_dims)
    mask_dim = (num, max_meta_Len)
    mask = np.zeros(mask_dim)
    for i, tensor in enumerate(sequences):
        length = len(tensor)
        tensor_arr = np.array(tensor)
        out_tensor[i, :length] = tensor_arr[:, :output_dim]
        mask[i, :length] = 1
    out_tensors.append(out_tensor)
    masks.append(mask)
    out_tensor = out_tensors[0]
    mask = masks[0]
    for i in range(1, len(out_tensors)):
        out_tensor = np.concatenate((out_tensor, out_tensors[i]), axis=0)
        mask = np.concatenate((mask, masks[i]), axis=0)
    return out_tensor, mask, max_meta_Len, traj_length


def get_option_from_seqs_lstm_meta(seqs_meta):
    #100 seqs; 1seq->time*traj*8; 1traj-> (s,a) pair 
    #time step of seq-.time stemp to LSTM
    #LSTM input: time stemp embedding*batchsize
    #output: time stemp*1*batchsize
    '''input:'''
    '''return: micro action/lstm'''
    #warm start for LSTM
    op_list_all = []
    op_list_meta = []
    for seqs in seqs_meta:
        for seq in seqs:
            op_list_temp = []
            l_temp = []
            for count,ti in enumerate(seq,start=1):
                #print(ti)
                l_temp.append(ti)
                if ti[0] == 3 and ti[1] == 6:
                    op = 0
                    op_list_temp += len(l_temp) * [op]
                    l_temp = []
                elif ti[0] == 6 and ti[1] == 2:
                    op = 1
                    op_list_temp += len(l_temp) * [op]
                    l_temp = []
                elif ti[0] == 7 and ti[1] == 9:
                    op = 2
                    op_list_temp += len(l_temp) * [op]
                    l_temp = []
                elif ti[0] == 10 and ti[1] == 6:
                    op = 3
                    op_list_temp += len(l_temp) * [op]
                    l_temp = []
                elif count == len(seq):
                    # Use the last confirmed option in this sequence
                    op_list_temp += len(l_temp) * [op]
                    l_temp = []
            op_list_all.append(op_list_temp)
        op_list_meta.append(op_list_all)
    return op_list_meta



def get_option_from_seqs_lstm_together(seqs_meta):
    #100 seqs; 1seq->time*traj*8; 1traj-> (s,a) pair 
    #time step of seq-.time stemp to LSTM
    #LSTM input: time stemp embedding*batchsize
    #output: time stemp*1*batchsize
    '''input:'''
    '''return: micro action/lstm'''
    #warm start for LSTM
    op_list_all = []
    op_list_meta = []
    for seqs in seqs_meta:
        #print(len(seqs_meta))
        for seq in seqs:
            op_list_temp = []
            l_temp = []
            for count,ti in enumerate(seq,start=1):
                #print(ti)
                l_temp.append(ti)
                if ti[0] == 3 and ti[1] == 6:
                    op = 0
                    op_list_temp += len(l_temp) * [op]
                    l_temp = []
                elif ti[0] == 6 and ti[1] == 2:
                    op = 1
                    op_list_temp += len(l_temp) * [op]
                    l_temp = []
                elif ti[0] == 7 and ti[1] == 9:
                    op = 2
                    op_list_temp += len(l_temp) * [op]
                    l_temp = []
                elif ti[0] == 10 and ti[1] == 6:
                    op = 3
                    op_list_temp += len(l_temp) * [op]
                    l_temp = []
                elif count == len(seq):
                    # Use the last confirmed option in this sequence
                    op_list_temp += len(l_temp) * [op]
                    l_temp = []
            op_list_all.append(op_list_temp)
    return op_list_all



def get_option_from_seqs_lstm_meta_task(seqs):
    #100 seqs; 1seq->time*traj*8; 1traj-> (s,a) pair 
    #time step of seq-.time stemp to LSTM
    #LSTM input: time stemp embedding*batchsize
    #output: time stemp*1*batchsize
    '''input:'''
    '''return: micro action/lstm'''
    #warm start for LSTM
    op_list_all = []
    op_list_meta = []
        #print(len(seqs_meta))
    for seq in seqs:
        op_list_temp = []
        l_temp = []
        for count,ti in enumerate(seq,start=1):
            #print(ti)
            l_temp.append(ti)
            if ti[0] == 3 and ti[1] == 6:
                op = 0
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 6 and ti[1] == 2:
                op = 1
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 7 and ti[1] == 9:
                op = 2
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 10 and ti[1] == 6:
                op = 3
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif count == len(seq):
                # Use the last confirmed option in this sequence
                op_list_temp += len(l_temp) * [op]
                l_temp = []
        op_list_all.append(op_list_temp)
    return op_list_all

def option_padding_together(sequences, max_len):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    print(num)
    out_dims = (num, max_len, 1)
    out_tensor = np.zeros(out_dims)
    for i, tensor in enumerate(sequences):
        length = len(tensor)
        tensor_arr = np.array(tensor).reshape(-1,1)
        out_tensor[i, :length] = tensor_arr
    return out_tensor


def option_padding_meta(meta, max_len):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    out_dims = (num, max_len, 1)
    out_tensor = np.zeros(out_dims)
    for i, tensor in enumerate(sequences):
        length = len(tensor)
        tensor_arr = np.array(tensor).reshape(-1,1)
        out_tensor[i, :length] = tensor_arr
    return out_tensor

def option_padding_meta(sequences_meta, max_len):
    """
    :param sequences: list of tensors
    :return:
    """
    out_tensors = []
    for sequences in sequences_meta:
        num = len(sequences)
        out_dims = (num, max_len, 1)
        out_tensor = np.zeros(out_dims)
        for i, tensor in enumerate(sequences):
            length = len(tensor)
            tensor_arr = np.array(tensor).reshape(-1,1)
            out_tensor[i, :length] = tensor_arr
        out_tensors.append(out_tensor)
    return out_tensors


def convert_traj_lstm_opt_arr_together(predicted_opt, input_traj_mask, traj_length):
    #convert the LSTM logits to opt via argmax
    predicted_opt = torch.argmax(predicted_opt, dim=2)
    mask_opt = input_traj_mask.view(-1)
    #np.savetxt("mask_opt.txt", mask_opt, delimiter=',', fmt='%d')
    predicted_opt = predicted_opt.view(-1)
    np.savetxt("option_arr.txt", predicted_opt, delimiter=',', fmt='%d')
   # print(predicted_opt)
    option_idx = torch.zeros(traj_length)
    init_opt_idx = 0
    #print(option_idx)
    for i in range(mask_opt.shape[0]):
        if mask_opt[i] == 1:
            #print(i, predicted_opt[i])
            option_idx[init_opt_idx] = predicted_opt[i]
            init_opt_idx+=1
    option_idx = Variable(option_idx.long())
    return option_idx

def traj_to_option_list_together(trajs):
    '''load traj from traj_file; seg/transfer traj to option idx; save option idx to option_file'''
    traj_arr = []
    for traj in trajs:
        print(traj.shape)
        for i in range(traj.shape[0]):
            for j in range(len(traj[i])):
                traj_arr.append(traj[i][j])
    traj_arr = np.array(traj_arr)
    return traj_arr

def get_option_from_seqs_lstm_six(seqs):
    #100 seqs; 1seq->time*traj*8; 1traj-> (s,a) pair 
    #time step of seq-.time stemp to LSTM
    #LSTM input: time stemp embedding*batchsize
    #output: time stemp*1*batchsize
    '''input:'''
    '''return: micro action/lstm'''
    #warm start for LSTM
    op_list_all = []
    op_list_meta = []
    for seq in seqs:
        op_list_temp = []
        l_temp = []
        for count,ti in enumerate(seq,start=1):
            #print(ti)
            l_temp.append(ti)
            if ti[0] == 2 and ti[1] == 5:
                op = 0
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 4 and ti[1] == 7:
                op = 1
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 7 and ti[1] == 5:
                op = 2
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 9 and ti[1] == 3:
                op = 3
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 9 and ti[1] == 6:
                op = 4
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif ti[0] == 12 and ti[1] == 4:
                op = 5
                op_list_temp += len(l_temp) * [op]
                l_temp = []
            elif count == len(seq):
                # Use the last confirmed option in this sequence
                op_list_temp += len(l_temp) * [op]
                l_temp = []
        op_list_all.append(op_list_temp)
    return op_list_all


