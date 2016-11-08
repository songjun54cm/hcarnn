__author__ = 'SongJun-Dell'
import os
import time
import cPickle as pickle
import numpy as np
import sys

from AttendContextSessionED import AttendContextSessionED
from solver.normal_solver import NormalSolver
from solver.parallel_normal_solver import ParallelNormalSolver
from data_provider.create_data_provider import create_dp
from init_parameters import init_params

def create_model(state, data_provider=None):
    t0 = time.time()

    if state['model_name'] == 'attend_context_session_ed':
        state = AttendContextSessionED.fullfill_state(state)
        model = AttendContextSessionED(state)
        if data_provider is not None:
            model.out_mapping_layer.b.setfield(np.array(data_provider.query_voca['query_prob']),
                                               model.out_mapping_layer.b.dtype)
    else:
        raise Exception('model name error')

    time_eclipse = time.time() - t0
    print 'model created in %f seconds.' % time_eclipse
    return model, state

def fullfill_default_state(state):
    state['data_set_name'] = state.get('data_set_name', 'toy_beijing').lower()
    state['model_name'] = state.get('model_name', 'normal_session_ed').lower()
    state['dp_name'] = state.get('dp_name', 'session_data_provider').lower()
    state['use_user'] = state.get('use_user', True)
    state['use_location'] = state.get('use_location', False)
    state['lstm_gate_act'] = 'sigmoid'
    state['lstm_hidden_act'] = 'tanh'
    state['batch_size'] = 500
    if state['parallel']:
        state['num_processes'] = 10
        state['batch_size'] = state.get('batch_size', 500)

    if state['model_name'] == 'attend_context_session_ed':
        if state['context_att_len'] is None: state['context_att_len'] = 3
    return state

def main(state):
    if state['test']:
        test_model(state)
        print 'Finish Test.'
    else:
        train_model(state)
        print 'Finish Train'

def train_model(state):
    state = fullfill_default_state(state)
    state = init_params(state)
    data_provider, state = create_dp(state)
    print 'state initialized.'
    # create model
    model_file_path = os.path.join(state['out_prefix'], 'model.pkl')
    state_file_path = os.path.join(state['out_prefix'], 'state.pkl')
    print 'create model...'
    model, state = create_model(state, data_provider)


    print 'creating solver...'
    solver = ParallelNormalSolver(state)

    print 'solver created.'

    print 'start training...'
    check_point_path = solver.train(model, data_provider, state)
    print 'training finish. start testing...'
    solver.test(model, data_provider)
    print 'Finish running!'


def test_model(state):
    print 'begin test'
    if state['model_name'] == 'normal_session_ed':
        model_folder = '/home/songjun/projects/map_query_suggestion_v2/output/normal_session_ed/character_beijing12highfreq3/out/4/check_point'
        model_files = ['model_checkpoint_normal_session_ed_character_beijing12highfreq3_mrr_0.427307.pkl',
                       'model_checkpoint_normal_session_ed_character_beijing12highfreq3_mrr_0.437557.pkl',
                       'model_checkpoint_normal_session_ed_character_beijing12highfreq3_mrr_0.448583.pkl',
                       'model_checkpoint_normal_session_ed_character_beijing12highfreq3_mrr_0.455191.pkl',
                       'model_checkpoint_normal_session_ed_character_beijing12highfreq3_mrr_0.458035.pkl',
                       'model_checkpoint_normal_session_ed_character_beijing12highfreq3_mrr_0.458988.pkl',
                       'model_checkpoint_normal_session_ed_character_beijing12highfreq3_mrr_0.460040.pkl',
                       'model_checkpoint_normal_session_ed_character_beijing12highfreq3_mrr_0.461637.pkl']
    elif state['model_name'] == 'normal_context_session_ed':
        model_folder = '/home/songjun/projects/map_query_suggestion_v2/output/normal_context_session_ed/character_beijing12highfreq3/out/3/check_point'
        model_files = ['model_checkpoint_normal_context_session_ed_character_beijing12highfreq3_mrr_0.430248.pkl',
                       'model_checkpoint_normal_context_session_ed_character_beijing12highfreq3_mrr_0.438566.pkl',
                       'model_checkpoint_normal_context_session_ed_character_beijing12highfreq3_mrr_0.449595.pkl',
                       'model_checkpoint_normal_context_session_ed_character_beijing12highfreq3_mrr_0.455012.pkl',
                       'model_checkpoint_normal_context_session_ed_character_beijing12highfreq3_mrr_0.459689.pkl',
                       'model_checkpoint_normal_context_session_ed_character_beijing12highfreq3_mrr_0.460694.pkl',
                       'model_checkpoint_normal_context_session_ed_character_beijing12highfreq3_mrr_0.462261.pkl',
                       'model_checkpoint_normal_context_session_ed_character_beijing12highfreq3_mrr_0.463411.pkl',
                       'model_checkpoint_normal_context_session_ed_character_beijing12highfreq3_mrr_0.464747.pkl']
    else:
        raise StandardError('model name error!')
    for mi, mfile in enumerate(model_files):
        print 'test model %s' % mfile
        model_path = os.path.join(model_folder, mfile)
        model_dump = pickle.load(open(model_path, 'rb'))
        state = model_dump['state']
        print 'model file loaded.'

        if mi == 0:
            state['log_file'] = os.path.join(state['out_folder'], 'test_log.log')
            data_provider, state = create_dp(state)
            model, state = create_model(state, data_provider)
            if state['parallel']:
                solver = ParallelNormalSolver(state)
            else:
                solver = NormalSolver(state)
            print 'solver created.'

        model.load_params(model_dump['params'])

        res = solver.validate_on_split(model, data_provider, split='train_valid')
        # validate on the validate data set
        res = solver.validate_on_split(model, data_provider, split='valid')
        # validate on the test data set
        res = solver.validate_on_split(model, data_provider, split='test')