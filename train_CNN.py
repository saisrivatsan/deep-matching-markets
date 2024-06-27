import os
import time
import logging
import argparse
import numpy as np
from typing import List

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_logger
from CNN import Net

from data import Data

device = "cuda"

class Args: 
    """ Env Params ------------------------------ """

    num_agents: int = 0
    """ Number of agents: To be filled later  """

    prob: float = 0.0
    """ Probability of truncation: To be filled later  """

    corr: float = 0.00
    """ Correlation Probability: To be filled later """

    lambd: float = 0.00
    """ Tradeoff param: To be filled later """

    """ Neural Network Params -------------------- """

    net_arch: List[int] = [64, 64, 64, 64, 64]
    """ Neural Network Architecture """

    act_fn = nn.LeakyReLU

    """ Optimization Params ---------------------- """

    batch_size: int = 256
    """ Batch size """
    
    num_accums: int = 4
    """ Num accumulation """

    learning_rate:  float = 5e-3
    """ Learning Rate """

    max_iteration: int = 50000
    """ Max iterations """

    """ Logging Params --------------------- """

    print_iter: int = 100
    val_iter: int = 1000
    """ Frequency of logging train stats, validation stats """

    save_iter: int = 1000
    """ Frequency of saving models """

    num_val_samples: int = 2560
    """ Validation batch """

    num_tst_samples: int = 20480
    """ Number of Test batches """

    """ Miscellaneous Params --------------------- """ 

    seed: int = 42
    """ Random Seed """
    
    resume: bool = False
    """ Start new (0) or resume (1) """
        

def torch_var(x): 
    return torch.Tensor(x).to(device)

""" Stability Violation """
def compute_st(r, p, q):
    num_agents = r.size(1)
    wp = F.relu(p[:, :, None, :] - p[:, :, :, None])
    wq = F.relu(q[:, :, None, :] - q[:, None, :, :])  
    t = (1 - torch.sum(r, dim = 1, keepdim = True))
    s = (1 - torch.sum(r, dim = 2, keepdim = True))
    rgt_1 = torch.einsum('bjc,bijc->bic', r, wq) + t * F.relu(q)
    rgt_2 = torch.einsum('bia,biac->bic', r, wp) + s * F.relu(p)
    regret =  rgt_1 * rgt_2 
    return regret.sum(-1).sum(-1).mean()/num_agents

""" IR Violation """
def compute_ir(r, p, q):
    num_agents = r.size(1)
    ir_1 = r * F.relu(-q)
    ir_2 = r * F.relu(-p)
    ir = ir_1 + ir_2
    return ir.sum(-1).sum(-1).mean()/num_agents

""" IC Violation for a single agent"""
def compute_ic_single(r, p, q, P, Q, agent_idx, is_P):       
    num_agents = r.size(1)
    P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx, is_P)
    p_mis, q_mis = torch_var(P_mis), torch_var(Q_mis)
    r_mis = model(p_mis.view(-1, num_agents, num_agents), q_mis.view(-1, num_agents, num_agents))
    r_mis = r_mis.view(*P_mis.shape)
    
    if is_P:
        r_diff = (r_mis[:, :, agent_idx, :] - r[:, None, agent_idx, :]) * (p[:, None, agent_idx, :] > 0).to(p.dtype)
        _, idx = torch.sort(-p[:, agent_idx, :])
                
    else:
        r_diff = (r_mis[:, :, :, agent_idx] - r[:, None, :, agent_idx]) * (q[:, None, :, agent_idx] > 0).to(q.dtype)
        _, idx = torch.sort(-q[:, :, agent_idx])
    
    idx = idx[:, None, :].repeat(1, r_mis.size(1), 1)
    fosd_viol = torch.cumsum(torch.gather(r_diff, -1, idx), -1)
    IC_viol = F.relu(fosd_viol).max(-1)[0].max(-1)[0].mean(-1)
    return IC_viol

""" IC Violation """
def compute_ic(r, p, q, P, Q):           
    num_agents = r.size(1)
    IC_viol_P = torch.zeros(num_agents).to(device)
    IC_viol_Q = torch.zeros(num_agents).to(device)
    
    for agent_idx in range(num_agents):
        IC_viol_P[agent_idx] = compute_ic_single(r, p, q, P, Q, agent_idx, is_P = True)
        IC_viol_Q[agent_idx] = compute_ic_single(r, p, q, P, Q, agent_idx, is_P = False)

    IC_viol = (IC_viol_P.mean() + IC_viol_Q.mean())/2
    return IC_viol


def evaluate(model, G, batch_size, num_samples):
    model.eval()
    num_batches = num_samples//batch_size
    with torch.no_grad():
        val_st_loss = 0.0
        val_ic_loss = 0.0
        for j in range(num_batches):
            P, Q = G.generate_batch(args.batch_size)
            p, q = torch_var(P), torch_var(Q)
            r = model(p, q)
            st_loss = compute_st(r, p, q)
            ic_loss = compute_ic(r, p, q, P, Q)
            val_st_loss += st_loss.item()/num_batches
            val_ic_loss += ic_loss.item()/num_batches
    return val_st_loss, val_ic_loss

                
if __name__ == "__main__":  
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-n', '--num_agents', action='store',
                    dest='num_agents', required=True, type=int,
                    help='Num Agents')
    
    parser.add_argument('-p', '--prob', action='store',
                    dest='prob', required=True, type=float,
                    help='Truncation Probability')
    
    parser.add_argument('-c', '--corr', action='store',
                    dest='corr', required=True, type=float,
                    help='Correlation Probability')
    
    parser.add_argument('-l', '--lambd', action='store',
                    dest='lambd', required=True, type=float,
                    help='Lambda')
    
    parser.add_argument('-r', '--resume', action='store',
                    dest='resume', default=0, type=int,
                    help='Resume Training')

    cmd_args = parser.parse_args()
    

    args = Args()
    args.num_agents = cmd_args.num_agents
    args.prob = cmd_args.prob
    args.corr = cmd_args.corr
    args.lambd = cmd_args.lambd
    args.resume = (cmd_args.resume == 1)
    
    """ Loggers """
    root_dir =  os.path.join("experiments", "agents_%d"%(args.num_agents),  "corr_%.2f"%(args.corr), "CNN") 
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    
    log_fname = os.path.join(root_dir, "LOG_lambd_%.4f"%(args.lambd))
    if args.resume:
        logger = init_logger(log_fname, filemode = 'a')
    else:
        logger = init_logger(log_fname)


    model_path = os.path.join(root_dir, "CHECKPOINT_lambd_%.4f"%(args.lambd))

    """ Seed for reproducibility """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    G = Data(args.num_agents, args.prob, args.corr) 
    model = Net(args.net_arch, args.act_fn).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10000,25000], gamma=0.5)
    
    iteration = 0
    
    if args.resume:
        checkpoint = torch.load(model_path)
        iteration = checkpoint['iteration']
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("*** Resuming Training ***")
          
    
    # Trainer
    tic = time.time()
    while iteration < args.max_iteration:

        # Reset opt
        opt.zero_grad()
        model.train()

        # Inference
        for _ in range(args.num_accums):
            P, Q = G.generate_batch(args.batch_size)    
            p, q = torch_var(P), torch_var(Q)
            r = model(p, q)

            # Compute loss
            st_loss = compute_st(r, p, q)

            if args.lambd < 1.0:
                ic_loss_p = compute_ic_single(r, p, q, P, Q, np.random.randint(args.num_agents), True)
                ic_loss_q = compute_ic_single(r, p, q, P, Q, np.random.randint(args.num_agents), False)
                ic_loss = (ic_loss_p + ic_loss_q)/2
            else:
                ic_loss = torch.tensor(0.0, device = device)

            total_loss = (st_loss * args.lambd + ic_loss * (1 - args.lambd))/args.num_accums
            total_loss.backward()

        opt.step()
        scheduler.step()
        t_elapsed = time.time() - tic

        iteration += 1

        # Validation  
        if iteration % args.print_iter == 0 or iteration == args.max_iteration:
            logger.info("[iter]: %d, [t]: %f, [stv]: %f, [rgt]: %f"%(iteration, t_elapsed, st_loss.item(), ic_loss.item()))

        if iteration % args.save_iter == 0 or iteration == args.max_iteration:
            checkpoint = dict()
            checkpoint['iteration'] = iteration
            checkpoint['model'] = model.state_dict()
            checkpoint['opt'] = opt.state_dict()
            checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, model_path)

        if iteration % args.val_iter == 0 or iteration == args.max_iteration:
            num_samples = args.num_tst_samples if iteration == args.max_iteration else args.num_val_samples
            val_st_loss, val_ic_loss = evaluate(model, G, args.batch_size, num_samples)
            logger.info("\t[TEST]: %d, [stv]: %f, [rgt]: %f"%(iteration, val_st_loss, val_ic_loss))
