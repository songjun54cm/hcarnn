__author__ = 'SongJun-Dell'
import platform
import os
from datetime import datetime
import cPickle as pickle
import time

def isWindows():
    sysstr = platform.system()
    if sysstr == 'Windows':
        return True
    else:
        return False

def get_proj_folder():
    if isWindows():
        return 'D:\\Projects\\query_suggestion\\map_query_suggestion_v2'
    else:
        p = '/root/data/projects/map_query_suggestion_v2/'
        if os.path.exists(p):
            return '/root/data/projects/map_query_suggestion_v2/'
        else:
            return '/home/songjun/projects/map_query_suggestion_v2/'

def parse_datetime(dtime):
    [d,t] = dtime.split(' ')
    year, mon, day = d.split('-')
    hour, mint, sec = t.split(':')
    return datetime(int(year), int(mon), int(day), int(hour), int(mint), int(sec))

def sort_dict(dict_count):
    dic = sorted(dict_count.iteritems(), key=lambda d:d[1], reverse=True)
    return dic

def insert_list_to_voca(in_list, voca=None):
    if voca is None:
        voca = {
            'w_to_ix': dict(),
            'ix_to_w': list(),
            'w_count': list(),
            'next_ix': 0
        }
    for key in in_list:
        voca['w_to_ix'][key] = voca['next_ix']
        voca['ix_to_w'].append(key)
        voca['w_count'].append(0)
        voca['next_ix'] += 1
    return voca

def get_voca_from_count(key_count, insert_list=[], unknow=True):
    if unknow:
        in_list = ['UNKNOW'] + insert_list
    else:
        in_list = insert_list
    voca = insert_list_to_voca(in_list)
    for k,v in key_count:
        voca['w_to_ix'][k] = voca['next_ix']
        voca['ix_to_w'].append(k)
        voca['w_count'].append(v)
        voca['next_ix'] += 1
    return voca

def get_raw_data_set_path(params):
    if params['data_set_name'] in ['beijing_12_highfreq3_win1',
                                   'beijing_12_highfreq3_win3',
                                   'beijing_12_highfreq3_win4',
                                   'beijing_12_highfreq3_win5',
                                   'beijing_12_highfreq3_session30minute_chunk3day',
        'beijing_12_highfreq3_session10minute_chunk3day',
        'beijing_12_highfreq3_session1hour_chunk3day',
        'beijing_12_highfreq3_session3hour_chunk3day',
        'beijing_12_highfreq3_session6hour_chunk3day',
        'beijing_12_highfreq3_session12hour_chunk3day',
                                   'character_beijing_12highfreq3_win2',
                                   'character_beijing_12highfreq3_win3',
                                   'character_beijing_12highfreq3_win4',
                                   'character_beijing_12highfreq3_win5']:
        print 'beijign_12_highfreq3'
        raw_data_path = os.path.join(params['proj_folder'], 'data', 'beijing_12_highfreq3', 'raw_data_set.pkl')
    elif params['data_set_name'] in ['toy_beijing', 'toy_beijing_session30minute_chunk3day']:
        print 'toy_beijing'
        raw_data_path = os.path.join(params['proj_folder'], 'data', 'toy_beijing', 'raw_data_set.pkl')
    else:
        print '%s' % params['data_set_name']
        raw_data_path = os.path.join(params['data_folder'], 'raw_data_set.pkl')
    return raw_data_path

def get_data_set(params):
    raw_data_set_path = get_raw_data_set_path(params)
    print 'get raw data set from %s' % raw_data_set_path
    with open(raw_data_set_path, 'rb') as f:
        data_set = pickle.load(f)
    return data_set

def counting_time(func):
    def _deco(*args, **kwargs):
        t0 = time.time()
        ret = func(*args, **kwargs)
        time_eclipse = time.time() - t0
        print 'run %s finish in %.3f seconds.' % (func.__name__, time_eclipse)
        return ret
    return _deco

def load_model(MODELCLASS, model_file_path):
    model_dump = pickle.load(open(model_file_path, 'rb'))
    state = model_dump['state']
    model = MODELCLASS(state)
    model.load_from_dump(model_dump)
    return model

def get_dp_data(data_set_name, dp_name):
    proj_folder = get_proj_folder()
    dp_file_path = os.path.join(proj_folder, 'data', data_set_name, dp_name+'.pkl')
    print 'loading data from %s' % dp_file_path
    dp_data = pickle.load(open(dp_file_path, 'rb'))
    return dp_data