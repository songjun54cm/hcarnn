__author__ = 'SongJun-Dell'
import os
from utils import get_data_set
from session_data_provider import SessDataProvider


def create_dp(params):
    if params['dp_name'] == 'session_data_provider':
        data_provider = SessDataProvider()
    dp_file_path = os.path.join(params['data_folder'], params['dp_name'] + '.pkl')
    if os.path.exists(dp_file_path):
        data_provider.load(dp_file_path)
        print 'load data provider from %s' % dp_file_path
    else:
        data_file_path = os.path.join(params['data_folder'], 'raw_data_set.pkl')
        data_set = get_data_set(params)
        data_provider.build(data_set)
        data_provider.save(dp_file_path)
        print 'build data provider and save to %s' % dp_file_path

    for p in data_provider.data_set_params.keys():
        params[p] = data_provider.data_set_params[p]
    params['cuid_unknow_idx'] = data_provider.cuid_voca['cuid_to_ix']['UNKNOW']
    params['city_unknow_idx'] = data_provider.city_voca['city_to_ix']['UNKNOW']
    params['query_unknow_idx'] = data_provider.query_voca['query_to_ix']['UNKNOW']
    params['query_start_idx'] = data_provider.query_voca['query_to_ix']['<START>']
    params['num_query'] = len(data_provider.query_voca['ix_to_query'])
    print 'data provider created!'

    return data_provider, params

def load_dp(params):
    if params['dp_name'] == 'session_data_provider':
        data_provider = SessDataProvider()
        dp_file_path = os.path.join(params['data_folder'], params['dp_name'] + '.pkl')
    if os.path.exists(dp_file_path):
        data_provider.load(dp_file_path)
        print 'load data provider from %s' % dp_file_path
    else:
        raise StandardError('data provider file not found')

    return data_provider