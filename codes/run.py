__author__ = 'SongJun-Dell'
import argparse
import os, sys
from models.run_normal import main as main_parallel_run

cur_folder = os.path.abspath('.')
sys.path.append(cur_folder)

def main(state):
    if state['model_name'] == 'attend_context_session_ed':
        main_parallel_run(state)
    else:
        raise StandardError('model name error!')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='data_set_name', type=str, default='beijing_12_highfreq3')
    parser.add_argument('-m', '--model', dest='model_name', type=str, default='attend_context_session_ed')
    parser.add_argument('--dp', '--data_provider', dest='dp_name', type=str, default='session_data_provider')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=500, help='batch size')
    parser.add_argument('-e', '--max_epoch', dest='max_epoch', type=int, default=10, help='epoch number')
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--grad_max_norm', dest='grad_max_norm', type=float, default=10.0, help='max gradient normalization')
    parser.add_argument('--valid_epoch', dest='valid_epoch', type=float, default=0.5, help='valid epoch')
    parser.add_argument('--regularize_rate', dest='regularize_rate', type=float, default=1e-8, help='regularize_rate')
    parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=10.0, help='grad_clip')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.0, help='momentum')
    parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.999, help='decay_rate')
    parser.add_argument('--sgd_mode', dest='sgd_mode', type=str, default='rmsprop')
    parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8)
    parser.add_argument('--cost_scale', dest='cost_scale', type=float, default=100)
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=128)
    parser.add_argument('--parallel', dest='parallel', type=int, default=1)
    parser.add_argument('--num_processes', dest='num_processes', type=int, default=0)
    parser.add_argument('--test', dest='test', type=int, default=0)
    parser.add_argument('--model_file', dest='model_file', type=str)
    parser.add_argument('--mode', dest='mode', type=str, default='train')
    parser.add_argument('--context_att_len', dest='context_att_len', type=int)

    args = parser.parse_args()
    state = vars(args)
    state['use_user'] = True
    state['use_location'] = False
    print 'use user: %s' % str(state['use_user'])
    print 'use location: %s' % str(state['use_location'])
    main(state)