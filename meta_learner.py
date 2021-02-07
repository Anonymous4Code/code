import torch
import torch.nn as nn
#import gym
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ExTrajectory import ExpertTraj
from utils import *
from modules import BC, Policy_meta, MLPEncoder, MLPDecoder, graph_VAE_inference, Segmentation, Actor_with_option, Discriminator
import os
import sys
import copy
from torch.autograd import Variable
#sys.path.append('%s/four_room' % os.path.dirname(os.path.realpath(__file__)))
import argparse
#from calculate_reward import calculate_reward
import higher
import pandas as pd
import collections
from sklearn.metrics import jaccard_score

######### Hyperparameters #########
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=0, help='Random seed.')
parser.add_argument('--CUDA', dest='CUDA', action='store_true',
                    help='enable CUDA.')
parser.add_argument('--CPU', dest='CUDA', action='store_false',
                    help='enable CUDA.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Initial learning rate.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='num of transitions sampled from expert.')
parser.add_argument('--graph_iter', type=int, default=1,
                    help='graph training epoch.')
parser.add_argument('--option_num', type=int, default=4,
                    help='option number.')
parser.add_argument('--state_dim', type=int, default=4,
                    help='dimension of state.')
parser.add_argument('--action_dim', type=int, default=4,
                    help='dimension of action.')   
parser.add_argument('--traj_iter', type=int, default=100,
                    help='training epoch of trajectory.')    
parser.add_argument('--policy_hidden_1', type=int, default=32,
                    help='hidden dim of pilicy network layer 1.')    
parser.add_argument('--policy_hidden_2', type=int, default=32,
                    help='hidden dim of pilicy network layer 2.') 
parser.add_argument('--option_dim', type=int, default=10,
                    help='option embedding dimension') 
parser.add_argument('--meta_train_epochs', type=int, default=50,
                    help='total training epochs') 
parser.add_argument('--adaptation_epochs', type=int, default=20,
                    help='total training epochs') 
parser.add_argument('--warm_start_epoch', type=int, default=1000,
                    help='LSTM warm start training epochs') 
parser.add_argument('--alpha', type=float, default=0.1,
                    help='graph aggregate info; range from 0 to 1')   
parser.add_argument('--meta_train_num', type=int, default=8,
                    help='meta train task number')  
parser.add_argument('--room_num', type=int, default=4,
                    help='room number')  
parser.add_argument('--demos', type=int, default=5,
                    help='demonstrations in each task')
parser.add_argument('--LSTM_seg_interval', type=int, default=5,
                    help='graph aggregate info; range from 0 to 1') 
parser.add_argument('--test_meta_task', type=int, default=5,
                    help='graph aggregate info; range from 0 to 1')                    
parser.add_argument('--save_folder', type=str, default='res/',
                    help='save vis result to folder ')  
parser.add_argument('--meta_iterator', type=int, default='20',
                    help='save vis result to folder ')     
parser.add_argument('--meta_lr', type=float, default='1e-3',
                    help='save vis result to folder ')      
parser.add_argument('--meta_inner', type=int, default='5',
                    help='save vis result to folder ')  
args = parser.parse_args()


def seg_pretrain(controller, meta_train_tasks, folder_name="four_room/generated_traj/"):
    '''LSTM warm start'''
    trajs = []
    traj_list = []
    traj_arr = []
    warm_start_optimizer = optim.Adam(list(controller.parameters()),
                       lr=1e-2, betas=(0.5, 0.999))

    for task_id in meta_train_tasks:
        traj_file_name = "data/demo_"+str(args.demos)+"_task_"+str(task_id)+".npy"
        traj = np.load(traj_file_name, allow_pickle = True)
        trajs.append(traj)
    traj_list = traj_to_option_list_together(trajs)
    traj_tensor = torch.Tensor(traj_list)
    input_traj_padded, input_traj_mask, padding_len, traj_length = traj_padding_together(trajs, args.state_dim)
    print("padding")
    option_all_file_name = "data/option_all_"+str(args.demos)+".npy"
    option_arrs = np.load(option_all_file_name, allow_pickle = True)
 
    target_opt_padded = option_padding_together(option_arrs, padding_len)
    input_traj_padded = torch.Tensor(input_traj_padded)
    target_opt_padded = torch.Tensor(target_opt_padded)
    input_traj_mask = torch.Tensor(input_traj_mask).unsqueeze(-1)
    LSTM_warm_start_criterion = nn.CrossEntropyLoss() #nn.NLLLoss()#

    if args.CUDA:
        target_opt_padded = target_opt_padded.cuda()
        input_traj_padded = input_traj_padded.cuda()
        input_traj_mask = input_traj_mask.cuda()
    '''LSTM warm start'''
    loss_warm_start_LSTM = []
    for warm_start in range(args.warm_start_epoch):
        indexes = np.random.randint(0, len(traj)*args.meta_train_num, size=args.batch_size)
        batch_mask = input_traj_mask[indexes]
        batch_traj_input = input_traj_padded[indexes]
        batch_opt_target = target_opt_padded[indexes]
        estimated_opt = controller(batch_traj_input)
        estimated_opt = estimated_opt*batch_mask
        estimated_opt = estimated_opt.permute(0, 2, 1)
        batch_masked_target_opt = batch_opt_target*batch_mask
        batch_masked_target_opt = batch_masked_target_opt.squeeze()
        warm_start_optimizer.zero_grad()
        loss_LSTM_warm_start = LSTM_warm_start_criterion(estimated_opt, batch_masked_target_opt.long())
        loss_LSTM_warm_start.backward()
        loss_warm_start_LSTM.append(loss_LSTM_warm_start)
        warm_start_optimizer.step()
    return controller


torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

loss_fn = nn.BCELoss()
loss_rec = nn.MSELoss()
controller = Segmentation(args.state_dim, args.policy_hidden_1, args.option_num) 
encoder = MLPEncoder(args.option_dim, 2*args.option_dim, 1, 0, True)
decoder = MLPDecoder(args.option_dim, 1, 2*args.option_dim, 2*args.option_dim, 2*args.option_dim, 0, False)
actor = Actor_with_option(args.state_dim, args.option_dim, args.action_dim)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(actor.parameters())+list(controller.parameters()),
                    lr=args.lr, betas=(0.5, 0.999))

discriminator = Discriminator(args.state_dim, args.action_dim)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
warm_start_optimizer = optim.Adam(list(controller.parameters()),
                       lr=1e-2, betas=(0.5, 0.999))

if args.CUDA:
    policy.cuda()
    encoder.cuda()
    decoder.cuda()
    controller.cuda()

meta_train_tasks = [0,1,2,4]
meta_test_tasks = [5,6]

hashmap = collections.defaultdict(dict)
option_hashmap = collections.defaultdict(dict)
x_state = pd.read_csv('x_mimic_meta_learning_with_task_label.csv')
y_action = pd.read_csv('y_mimic_meta_learning_option_number6_old_action.csv')  
patient_matrix = x_state.values
action_matrix = y_action.values
option_matrix = copy.deepcopy(action_matrix)

prints = []
for i in range(len(patient_matrix)):
    patient_id = int(patient_matrix[i, 0])
    task_id = int(patient_matrix[i, 1])
    patient_value = patient_matrix[i, 2:]
    option = option_matrix[i, 1]
    action_matrix = np.where(action_matrix == 0, action_matrix, 1)
    action_value = action_matrix[i, 2:]
    state_action = np.concatenate([patient_value, action_value])
    if patient_id not in hashmap[task_id]:
        hashmap[task_id][patient_id] = []
        option_hashmap[task_id][patient_id] = []
    hashmap[task_id][patient_id].append(state_action)
    option_hashmap[task_id][patient_id].append(option)
tasks_hashmap = collections.defaultdict(dict)
for task_id in hashmap:
    shot_id = 0
    for patient in hashmap[task_id]:
        if 30>=len(hashmap[task_id][patient])>=10:
            if shot_id < args.demos:
                tasks_hashmap[task_id][patient] = copy.deepcopy(hashmap[task_id][patient])
                shot_id +=1 

option_all = []
for task_id in meta_train_tasks:
    traj_file_name = "data/demo_"+str(args.demos)+"_task_"+str(task_id)+".npy"
    option_file_name = "data/option_"+str(args.demos)+"_task_"+str(task_id)+".npy"
    cur_demo_traj = [] 
    option_traj = []
   # print(len(tasks_hashmap[task_id]))
    for patient in tasks_hashmap[task_id]:
        cur_demo_traj.append(tasks_hashmap[task_id][patient])
        option_traj.append(option_hashmap[task_id][patient])
        option_all.append(option_hashmap[task_id][patient])
    np.save(traj_file_name, np.array(cur_demo_traj))
    np.save(option_file_name, np.array(option_traj))
option_all_file_name = "data/option_all_"+str(args.demos)+".npy"
np.save(option_all_file_name, np.array(option_all))

#************************** Pretrain LSTM ****************************
controller = seg_pretrain(controller, meta_train_tasks)
#tasks -> demonstrations
#************************** LOAD DATA ****************************
trajs = []
trajs_list = []

for task_id in meta_train_tasks:
    traj_file_name = "data/demo_"+str(args.demos)+"_task_"+str(task_id)+".npy"
    traj = np.load(traj_file_name, allow_pickle = True)
   # print(traj)
    trajs.append(traj)
    traj_list = traj_to_option_list_meta_task(traj)
    trajs_list.append(traj_list)

option_arrs = []
trajs_length = []
input_trajs_padded = []
input_trajs_mask = []
for i in range(args.meta_train_num):
    #print(args.meta_train_num)
   # print(len(trajs))
    input_traj_padded, input_traj_mask, padding_len, traj_length = traj_padding_meta_task(trajs[i], args.state_dim)
    #option_arr = np.array(get_option_from_seqs_lstm_meta_task(trajs[i])) 
    input_traj_padded = torch.Tensor(input_traj_padded)
    input_traj_mask = torch.Tensor(input_traj_mask).unsqueeze(-1)
    predicted_opt = controller(input_traj_padded)
    option_arr = convert_traj_lstm_opt_arr_together(predicted_opt, input_traj_mask, traj_length)
    option_arrs.append(option_arr)
    trajs_length.append(traj_length)
    input_trajs_padded.append(input_traj_padded)
    input_trajs_mask.append(input_traj_mask)

#get option embedding and graph input
option_emb = random_init_option(args.option_num, args.option_dim) # random initial the option embedding
option_emb = torch.tensor(option_emb).float()
option_emb = Variable(option_emb) #option emb to variable
input_rel_rec, input_rel_send = get_graph_rec_send(args.option_num)
if args.CUDA:
    input_rel_rec = input_rel_rec.cuda()
    input_rel_send = input_rel_send.cuda()
rel_rec = input_rel_rec
rel_send = input_rel_send

#************************** META TRAIN ****************************
#tf_logger = Logger(os.path.join(sys.path[0]+'/tensorboard/'))
qry_losses = []
res_to_file = []
vis_mi_aucs = []
vis_ma_aucs = []
discriminator_iters = 5
discriminator_loss = []
KL_loss = []
generator_loss = []
for epoch in range(args.meta_train_epochs):
    print("epoch", epoch)
    if args.CUDA:
        option_emb = option_emb.cuda()
    if epoch>40:
        for i in range(args.graph_iter):
            input_rel_rec, input_rel_send = get_graph_rec_send(args.option_num)
            if args.CUDA:
                input_rel_rec = input_rel_rec.cuda()
                input_rel_send = input_rel_send.cuda()
            last_epoch_option_emb = option_emb
            input_rel_rec = (1-args.alpha)*input_rel_rec + (args.alpha)*rel_rec
            input_rel_send = (1-args.alpha)*input_rel_send + (args.alpha)*rel_send
            option_emb, rel_rec, rel_send, cur_implicit_graph = graph_VAE_inference(encoder, decoder, last_epoch_option_emb, input_rel_rec, input_rel_send, args.option_num, args.CUDA)
            optimizer.zero_grad()
            loss_graph_rec = loss_rec(option_emb, last_epoch_option_emb)
            optimizer.step()

    d_options = []
    for i in range(args.meta_train_num):
        d_option = torch.zeros(trajs_length[i], args.option_dim)
        for j in range(trajs_length[i]):
            d_option[j:] = option_emb[option_arrs[i][j]]
        if args.CUDA:
            d_option = d_option.cuda()
        d_options.append(d_option)

    experts = []

    for i in range(args.meta_train_num):
        traj_tensor_task  = torch.Tensor(trajs_list[i])
        if args.CUDA:
            traj_tensor_task = traj_tensor_task.cuda()
        experts.append(ExpertTraj(traj_tensor_task, d_options[i], args.state_dim, 0.8))

    inner_opt = torch.optim.SGD(actor.parameters(), lr=args.meta_lr)
    optimizer.zero_grad()
    cur_loss = []
    cur_mi = []
    cur_ma = []
    cur_loss = []
    task_loss_KL = []
    task_discriminator = []
    task_generator = []
    max_jaccard = 0
    for jaccard_threshold in [0.035]:
      #  max_jaccard = 0
        for _ in range(args.meta_iterator):
            for i in range(args.meta_train_num):
                with higher.innerloop_ctx(actor, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                    for _ in range(args.meta_inner):
                        exp_state, exp_action, expert_option = experts[i].sample(args.batch_size)
                        # sample expert states for actor
                        state, _, option_traj = experts[i].sample(args.batch_size)
                        action = fnet(state, option_traj)
                        loss_actor = -discriminator(state, action)
                        diffopt.step(loss_actor.mean())

                exp_state, exp_action, _ = experts[i].sample(args.batch_size)
                # sample expert states for actor
                state, _, option = experts[i].sample(args.batch_size)
                action = fnet(state, option)
                # if i%20==0:
                optimizer_discriminator.zero_grad()
                # label tensors
                exp_label= torch.full((args.batch_size,1), 1)
                policy_label = torch.full((args.batch_size,1), 0)
                prob_exp = discriminator(exp_state, exp_action)
                loss = F.binary_cross_entropy(prob_exp, exp_label)
                loss_kl = F.binary_cross_entropy(prob_exp, exp_label)
                task_loss_KL.append(loss_kl)
                # with policy transitions
                prob_policy = discriminator(state, action.detach())
                loss += F.binary_cross_entropy(prob_policy, policy_label)
                task_discriminator.append(loss)
                # take gradient step
                loss.backward()
                optimizer_discriminator.step()

                inner_generator = 1
                for _ in range(inner_generator):
                    state, _, option = experts[i].sample(args.batch_size)
                    action = fnet(state, option)
                    optimizer.zero_grad()
                    loss_actor = -discriminator(state, action)
                    loss_actor.mean().backward(retain_graph=True)
                    optimizer.step()
                task_generator.append(loss_actor.mean())
                val_state, val_action, val_option = experts[i].eval()
                mi_auc, actions_evl_masked, actions_pre_masked = evaluation(fnet, val_state, val_action, val_option, args.action_dim)
                
                cur_mi.append(mi_auc)
                if epoch > 900:
                    ori_actions_pre_masked = copy.deepcopy(actions_pre_masked)
                    for patient_ID in range(len(actions_evl_masked)):
                        cur = actions_pre_masked[patient_ID]
                        cur[cur>jaccard_threshold] = 1
                        cur[cur<=jaccard_threshold] = 0
                        actions_pre_masked[patient_ID] = cur
                    print(actions_evl_masked, actions_pre_masked)
                    if jaccard_score(actions_evl_masked, actions_pre_masked, average='samples') > max_jaccard:
                        max_jaccard = jaccard_score(actions_evl_masked, actions_pre_masked, average='samples')
                        np.save("eval", actions_evl_masked)
                        np.save("pred", ori_actions_pre_masked)
        KL_loss.append(sum(task_loss_KL) / (args.meta_train_num * args.meta_iterator)) 
        discriminator_loss.append(sum(task_discriminator) / (args.meta_train_num * args.meta_iterator))  
        generator_loss.append(sum(task_generator) / (args.meta_train_num * args.meta_iterator))
        print(max_jaccard, jaccard_threshold)
    evaluate_res = [epoch]+[sum(cur_mi) / (args.meta_train_num * args.meta_iterator), np.std(cur_mi)]
    res_to_file.append(evaluate_res)

    if epoch>40:
        if args.graph_iter>0:
            graph_file = "meta_gail/"+str(args.meta_iterator)+"_lr_"+str(args.meta_lr)+"_inner_"+str(args.meta_inner)+"_res"+"implicit_graph.txt"
            np.savetxt(graph_file, cur_implicit_graph, delimiter=',', fmt='%.3f')
           # visualize_graph_option(args.save_folder, cur_implicit_graph, option_emb, epoch)

#************************** ZERO SHOT ****************************
para_file = args.save_folder+"zero_shot_res"+str(args.meta_iterator)+"_" +str(args.meta_inner)+"_"+ str(inner_generator)+".txt"
f=open(para_file,'ba')
np.savetxt(f, res_to_file, delimiter=',', fmt=['%d','%.5f', '%.5f'])\
#rewards_vis.append(reward_mean)

output_fig = args.save_folder+"Loss_KL"+str(args.meta_iterator)+"_" +str(args.meta_inner)+"_"+ str(inner_generator)+".png"
plt.plot(KL_loss)
plt.grid(True)
plt.xlabel('Training epoch')
plt.ylabel('Cross entropy loss')
plt.title('Cross entropy of generated and experts')
plt.show(block=False)
plt.savefig(output_fig, format="PNG")
plt.close()

output_fig = args.save_folder+"Discriminator"+str(args.meta_iterator)+"_" +str(args.meta_inner)+"_"+ str(inner_generator)+".png"
plt.plot(discriminator_loss)
plt.grid(True)
plt.xlabel('Training epoch')
plt.ylabel('Cross entropy loss')
plt.title('Cross entropy loss of discriminator')
plt.show(block=False)
plt.savefig(output_fig, format="PNG")
plt.close()

output_fig = args.save_folder+"Generator"+str(args.meta_iterator)+"_" +str(args.meta_inner)+"_"+ str(inner_generator)+".png"
plt.plot(generator_loss)
plt.grid(True)
plt.xlabel('Training epoch')
plt.ylabel('Likelihood')
plt.title('Loss of generator')
plt.show(block=False)
plt.savefig(output_fig, format="PNG")
plt.close()

