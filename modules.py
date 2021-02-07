import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

class Policy(nn.Module):
    def __init__(self, state_dim, emb_dim, hidden1, hidden2, action_dim):
        super(Policy, self).__init__()

        self.l1_emb = nn.Linear(emb_dim + state_dim, emb_dim + state_dim)
        self.linear_in = nn.Linear(state_dim, state_dim, bias=False)
        self.linear_out = nn.Linear(emb_dim + state_dim , hidden1, bias=False)
        self.linear_out_2 = nn.Linear(state_dim+emb_dim, hidden1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.l1 = nn.Linear(hidden1, hidden2)
        self.l2 = nn.Linear(hidden2, action_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        #self.attention = attention

    def forward(self, state, o_emb):
        combined = torch.cat((state, o_emb), dim=1)
        #combined = state #not use option information
        output = self.linear_out_2(combined)
        output = self.tanh(output)
        x = self.relu(self.l1(output))
        x = self.sigmoid(self.l2(x))
        return x

class Policy_meta(nn.Module):
    def __init__(self, state_dim, emb_dim, hidden1, hidden2, action_dim):
        super(Policy_meta, self).__init__()

        self.l1_emb = nn.Linear(emb_dim + state_dim, emb_dim + state_dim)
        self.linear_in = nn.Linear(state_dim, state_dim, bias=False)
        self.linear_out = nn.Linear(emb_dim + state_dim , hidden1, bias=False)
        self.linear_out_2 = nn.Linear(state_dim+emb_dim, hidden1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.l1 = nn.Linear(hidden1, hidden2)
        self.l2 = nn.Linear(hidden2, action_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        #self.attention = attention

    def forward(self, state, o_emb):
        combined = torch.cat((state, o_emb), dim=1)
        #combined = state #not use option information
        output = self.linear_out_2(combined)
        output = self.tanh(output)
        x = self.relu(self.l1(output))
       # x = self.sigmoid(self.l2(x))
        x = self.l2(x)
        return x

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs
        x = self.bn(x)
        return x

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_dims]
        x = inputs
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        #x)
        x = self.mlp1(x)  # 2-layer ELU net per node
        #print(rel_rec.size())
        x = self.node2edge(x, rel_rec, rel_send)

        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=1)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=1)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)

class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.Linear(2 * n_in_node, msg_hid)
        self.msg_fc2 = nn.Linear(msg_hid, msg_out)
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send):
        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=1)
        #print("pre_msg")
        #print(pre_msg.size())
        all_msgs = Variable(torch.zeros(pre_msg.size(0), self.msg_out_shape))
        #print("all_msgs")
        #print(all_msgs.size())
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        
        msg = F.relu(self.msg_fc1(pre_msg))
        msg = F.dropout(msg, p=self.dropout_prob)
        msg = F.relu(self.msg_fc2(msg))
        #print(msg.size())
        all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.
        #preds = []
        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs
        last_pred = self.single_step_forward(last_pred, rel_rec, rel_send)
        #preds.append(last_pred)

        sizes = [last_pred.size(0), last_pred.size(1)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        output = last_pred
        return output
        #pred_all = output[:, :(inputs.size(1) - 1), :, :]
        #return pred_all.transpose(1, 2).contiguous()

class Segmentation(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer=1):
        #state dim; 1;
        super(Segmentation, self).__init__()
        #batch, time stemp, state dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True) #bidirectional=True, batch_first=True
        self.out_fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        x, state = self.lstm(input)
        out = self.out_fc(x)
        #out = self.softmax(out)
        return out

class MLP_Segmentation(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        #state dim; 1;
        super(MLP_Segmentation, self).__init__()
        #batch, time stemp, state dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        x = self.fc1(input)
        out = self.fc2(x)
        #out = self.softmax(out)
        return out

def graph_VAE_inference_New(encoder, decoder, option_emb, input_rel_rec, input_rel_send, option_num, CUDA = False):
    logits = encoder(option_emb, input_rel_rec, input_rel_send)
    graph_logits = logits.data
    graph_logits = graph_logits.resize_(option_num,option_num-1).cpu()
    cur_implicit_graph = convert_to_graph(graph_logits.numpy(), option_num)
    rel_rec_np = np.array(encode_onehot(np.where(cur_implicit_graph)[0]), dtype=np.float32) #(N)*(N-1)
    rel_send_np = np.array(encode_onehot(np.where(cur_implicit_graph)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec_np)
    rel_send = torch.FloatTensor(rel_send_np)
    if CUDA:
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()
    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)
    #print(rel_rec.size())
    #print(rel_send.size())
    option_emb_updated = decoder(option_emb, rel_rec, rel_send)
    return option_emb_updated, rel_rec, rel_send, cur_implicit_graph

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        
        
    def forward(self, x):
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x

class Actor_with_option(nn.Module):
    def __init__(self, state_dim, emb_dim, action_dim):
        super(Actor_with_option, self).__init__()
        
        self.l1 = nn.Linear(emb_dim + state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        
        
    def forward(self, state, o_emb):
        combined = torch.cat((state, o_emb), dim=1)
        x = F.elu(self.l1(combined))
        x = F.elu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        
        self.l1 = nn.Linear(state_dim+action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        # print('state',state.shape)
        # print('action', action.shape)
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x


class BC():
    def __init__(self, state_dim, emb_dim, hidden1, hidden2, action_dim, lr, betas, ex_buffer):
        self.policy = Policy(state_dim, emb_dim, hidden1, hidden2, action_dim)
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.loss_fn = nn.BCELoss()
    def select_action(self, state, ops):
        actions = self.policy.forward(state, ops)
        return actions

    def update(self, n_iter, batch_size=100):
        for i in range(n_iter):
            state_traj, action_traj, option_traj = self.expert.sample(batch_size)
            state_traj = torch.FloatTensor(state_traj)
            action_traj = torch.FloatTensor(action_traj)
            option_traj = torch.FloatTensor(option_traj)
            #### policy loss #####
            actions = self.select_action(state_traj, option_traj)
            policy_loss = self.loss_fn(actions, action_traj)
            self.optim_policy.zero_grad()
            policy_loss.backward(retain_graph = True)
            self.optim_policy.step()


class GAIL:
    def __init__(self, env_name, state_dim, action_dim, lr, betas):
        self.actor = Actor(state_dim, action_dim)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=betas)
        
        self.discriminator = Discriminator(state_dim, action_dim)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.expert = ExpertTraj(env_name)
        self.loss_fn = nn.BCELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        return self.actor(state).cpu().data.numpy()
        
    def update(self, n_iter, batch_size=100):

        discriminator_loss = []
        for i in range(n_iter):
            # sample expert transitions
            exp_state, exp_action,_ = self.expert.sample(batch_size)
            exp_state = torch.FloatTensor(exp_state)
            exp_action = torch.FloatTensor(exp_action)   
            # sample expert states for actor
            state, _, _ = self.expert.sample(batch_size)
            state = torch.FloatTensor(state)
            action = self.actor(state)
            # if i%20==0:
            self.optim_discriminator.zero_grad()
            # label tensors
            exp_label= torch.full((batch_size,1), 1)
            policy_label = torch.full((batch_size,1), 0)
            prob_exp = self.discriminator(exp_state, exp_action)
            loss = self.loss_fn(prob_exp, exp_label)
            loss_a = loss
            # with policy transitions
            prob_policy = self.discriminator(state, action.detach())
            loss += self.loss_fn(prob_policy, policy_label)

            discriminator_loss.append(loss)
            # take gradient step
            loss.backward()
            self.optim_discriminator.step()
            self.optim_actor.zero_grad()
            loss_actor = -self.discriminator(state, action)
            loss_actor.mean().backward()
            self.optim_actor.step()


        print('loss',loss)
        return discriminator_loss