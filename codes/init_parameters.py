__author__ = 'SongJun-Dell'
import os
from utils import get_proj_folder

def init_params(params):
    params['data_set_name'] = params['data_set_name'].lower()
    params['proj_folder'] = get_proj_folder()
    params['data_folder'] = os.path.join(params['proj_folder'], 'data', params['data_set_name'])
    params = get_out_folder(params)

    params['train_log_file'] = os.path.join(params['out_prefix'], 'train_log.log')
    params['valid_log_file'] = os.path.join(params['out_prefix'], 'valid_log.log')

    if params['model_name'] == 'attend_context_session_ed':
        params = init_params_normal_context_session_ed(params)

        params['batch_size'] = params.get('batch_size', 50)
        params['max_epoch'] = params.get('max_epoch', 30)
        params['valid_epoch'] = params.get('valid_epoch', 0.5)
        params['valid_batch_size'] = params.get('valid_batch_size', params['batch_size'])
        params['regularize_rate'] = params.get('regularize_rate', 1e-8)
        params['grad_clip'] = params.get('grad_clip', 10)
        params['grad_max_norm'] = params.get('grad_max_norm', 10)
        params['momentum'] = params.get('momentum', 0.0)
        params['decay_rate'] = params.get('decay_rate', 0.999)
        params['learning_rate'] = params.get('learning_rate', 1e-8)
        params['sgd_mode'] = params.get('sgd_mode', 'rmsprop')
        params['smooth_eps'] = params.get('smooth_eps', 1e-8)
        params['cost_scale'] = params.get('cost_scale', 100)
        params['loss_scale'] = params.get('loss_scale', 100)
    print 'params init finish!'
    return params

def get_out_folder(params):
    if not params.has_key('out_folder'):
        num = 1
        out_prefix = os.path.join(params['proj_folder'], 'output', params['model_name'], params['data_set_name'], 'out')
        params['out_prefix'] = out_prefix
        out_folder = os.path.join(out_prefix, str(num))
        while os.path.exists(out_folder):
            num += 1
            out_folder = os.path.join(out_prefix, str(num))
        os.makedirs(out_folder)
        params['out_folder'] = out_folder
        params['log_file'] = os.path.join(out_folder, 'log.log')
        params['checkpoint_out_dir'] = os.path.join(out_folder, 'check_point')
        os.makedirs(params['checkpoint_out_dir'])
    return params

def init_params_NormalSessionED(params):
    params['hidden_size'] = params.get('hidden_size', 32)
    params['cuid_emb_size'] = params.get('cuid_emb_size', params['hidden_size'])
    params['city_emb_size'] = params.get('city_emb_size', params['hidden_size'])
    params['query_emb_size'] = params.get('query_emb_size', params['hidden_size'])
    params['encode_hidden_size'] = params.get('encode_hidden_size', params['hidden_size'])
    params['session_hidden_size'] = params.get('session_hidden_size', params['hidden_size'])
    params['decode_hidden_size'] = params.get('decode_hidden_size', params['hidden_size'])
    return params

def init_params_normal_context_session_ed(params):
    return init_params_NormalSessionED(params)
