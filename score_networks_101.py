import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from scores import get_score_func, get_auc_score_func, get_score_sum_func
from scipy import stats
from pycls.models.nas.nas import Cell
from utils import add_dropout, init_network

from sklearn.metrics import roc_auc_score as auc

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='nasbench_only108.tfrecord',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench101', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', default='trainval', action='store_true')

parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int,
                    help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    # y 128 1 out 128 64 jacob 128 3 32 32
    return jacob, target.detach(), y.detach(), out.detach()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
savedataset = args.dataset
dataset = 'fake' if 'fake' in args.dataset else args.dataset
args.dataset = args.dataset.replace('fake', '')
if args.dataset == 'cifar10':
    args.dataset = args.dataset + '-valid'
searchspace = nasspace.get_search_space(args)
if 'valid' in args.dataset:
    args.dataset = args.dataset.replace('-valid', '')
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat,
                                 args)
os.makedirs(args.save_loc, exist_ok=True)

filename = f'{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{savedataset}{"_" + args.init + "_" if args.init != "" else args.init}_{"_dropout" if args.dropout else ""}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}'
accfilename = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{savedataset}_{args.trainval}'

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'


amount = 8000  # len(searchspace)
scores = np.zeros(len(searchspace))
scores_auc = np.zeros(len(searchspace))
scores_sum = np.zeros(len(searchspace))

try:
    accs = np.load(accfilename + '.npy')
except:
    accs = np.zeros(len(searchspace))

counter = -1
for i, (uid, network) in enumerate(searchspace):
    # Reproducibility
    try:
        if args.dropout:
            add_dropout(network, args.sigma)
        if args.init != '':
            init_network(network, args.init)
        if 'hook_' in args.score:
            network.K = np.zeros((args.batch_size, args.batch_size))


            def counting_forward_hook(module, inp, out):
                try:
                    if not module.visited_backwards:
                        return
                    if isinstance(inp, tuple):
                        inp = inp[0]
                    inp = inp.view(inp.size(0), -1)
                    x = (inp > 0).float()
                    K = x @ x.t()
                    K2 = (1. - x) @ (1. - x.t())
                    network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
                    #    有不同加一， 所以k越大， 相似度越低
                except:
                    pass


            def counting_backward_hook(module, inp, out):
                module.visited_backwards = True


            for name, module in network.named_modules():
                if 'ReLU' in str(type(module)):
                    # hooks[name] = module.register_forward_hook(counting_hook)
                    module.register_forward_hook(counting_forward_hook)
                    module.register_backward_hook(counting_backward_hook)

        network = network.to(device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        s = []
        s_auc = []
        s_sum = []
        for j in range(args.maxofn):
            data_iterator = iter(train_loader)
            x, target = next(data_iterator)
            # x 128 3 32 32 图片数据， target 0-9 类别数据
            x2 = torch.clone(x)
            x2 = x2.to(device)
            x, target = x.to(device), target.to(device)
            jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)

            if 'hook_' in args.score:
                network(x2.to(device))
                s.append(get_score_func(args.score)(network.K, target))
                s_auc.append(get_auc_score_func(network.K, target))
                s_sum.append(get_score_sum_func(network.K))
            else:
                s.append(get_score_func(args.score)(jacobs, labels))

        if 1:
            counter += 1
            scores[counter] = np.mean(s)
            scores_auc[counter] = np.mean(s_auc)
            scores_sum[counter] = np.mean(s_sum)

            accs[counter] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
            if i % 100 == 0:
                accs_ = accs[~np.isnan(scores)]

                scores_ = scores[~np.isnan(scores)]
                scores_auc_ = scores_auc[~np.isnan(scores_auc)]
                scores_sum_ = scores_sum[~np.isnan(scores_sum)]

                numnan = np.isnan(scores).sum()
                numnan_auc = np.isnan(scores_auc).sum()
                numnan_sum = np.isnan(scores_sum).sum()

                tau, p = stats.kendalltau(accs_[:max(counter - numnan, 1)], scores_[:max(counter - numnan, 1)])
                tau_auc, p_auc = stats.kendalltau(accs_[:max(counter - numnan_auc, 1)],
                                                  scores_auc_[:max(counter - numnan_auc, 1)])
                tau_sum, p_sum = stats.kendalltau(accs_[:max(counter - numnan_sum, 1)],
                                                  scores_sum_[:max(counter - numnan_sum, 1)])

                print(i, counter, f'{tau}', f'{tau_sum}', f'{tau_auc}', accs[counter], scores[counter], scores_sum[counter], scores_auc[counter])

            if counter % amount == 0 and counter > 0:
                print(np.sum(scores) / amount, np.sum(accs[:amount]) / amount, np.sum(scores_sum) / amount, np.sum(scores_auc) / amount)

                np.save('results/201_s.npy', scores)
                np.save('results/201_s_sum.npy', scores_sum)
                np.save('results/201_s_auv.npy', scores_auc)
                np.save('results/201_acc.npy', accs)
                exit()
        else:
            continue


    except Exception as e:
        print(e)
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        scores[i] = np.nan
np.save(filename, scores)
np.save(accfilename, accs)
