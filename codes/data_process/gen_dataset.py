import argparse
import json
import os
import cPickle as pickle
import time
import sys
from utils import get_proj_folder, parse_datetime
__author__ = 'SongJun-Dell'

def clean_key(query_key, split_characters):
    for s in split_characters:
        query_key = query_key.split(s)[0]
    return query_key

def get_split_characters(params):
    params['split_characters_path'] = params.get('split_characters_path', os.path.join(params['data_folder'], 'split_characters'))
    lines = open(params['split_characters_path'], 'rb').readlines()
    split_characters =[' '] + [s.strip() for s in lines if s != ' \n']
    print '%d split characters: %s' % (len(split_characters), repr(split_characters))
    return split_characters

def get_stop_querys(params):
    params['stop_queries_path'] = params.get('stop_queries_path', os.path.join(params['data_folder'], 'stop_queries'))
    lines = open(params['stop_queries_path'], 'rb').readlines()
    stop_querys = [s.strip() for s in lines] + [None]
    print '%d stop queries: %s ' % (len(stop_querys), repr(stop_querys))
    return stop_querys

def get_data_set_conf(params):
    params['data_folder'] = os.path.join(params['proj_folder'], 'data', params['data_set_name'])
    if not os.path.exists(params['data_folder']):
        os.makedirs(params['data_folder'])
        print 'create folder %s' % params['data_folder']
    params['split_characters_path'] = os.path.join(params['proj_folder'], 'data', 'split_characters')
    params['stop_queries_path'] = os.path.join(params['proj_folder'], 'data', 'stop_queries')

    if params['data_set_name'] == 'toy_beijing':
        params['raw_data_file_path'] = os.path.join(params['data_folder'], 'beijing=1510_navi_000010_0')
    elif params['data_set_name'] == 'beijing_12_highfreq3':
        params['raw_data_file_path'] = os.path.join(params['data_folder'], 'high_frequency_data3')

    return params

def find_appropriate_position(query_list, query):
    pos = -1
    cur_time = query['time_stamp']

    qend = -len(query_list)
    while pos > qend:
        prev_time = query_list[pos]['time_stamp']
        if cur_time < prev_time:
            pos -= 1
        else:
            pos += 1
            break
    assert pos < 0, 'pos error: %d' % pos
    return pos

def gen_data_set(params):
    """
    generate data set, queries in the query list are sorted in time order.
    :param params:
    :return: data_set = {
        'map_query_data' = { cuid: query_list }
    }
        query_list = [ query, ... ]
            query = {
                'time_stamp': time stamp,
                'dtime': date time,
                'loc_city': locate city,
                'query_key': query key,
                'query_loc': query locate city
                'ftype': function type,
                'plist': poi list
            }
    """
    params = get_data_set_conf(params)
    split_characters = get_split_characters(params)
    stop_querys = get_stop_querys(params)

    failed_query_count = 0.0
    query_history_count = 0.0

    raw_data_file = open(params['raw_data_file_path'], 'rb')

    map_query_data = dict()
    line_count = 0
    t0 = time.time()
    line = raw_data_file.readline()
    while line:
        line_count += 1
        if line_count % 10000 == 0:
            time_eclipse = time.time() - t0
            print '%d lines processed in %f seconds!' % (line_count, time_eclipse)
            t0 = time.time()
        try:
            [cuid, mb, hiss] = line.split('\x01')
        except:
            print 'error at line %d' % line_count
            print line
            sys.exit(0)

        his_list = hiss.split('\x02')
        for his in his_list:
            query_history_count += 1
            [dtime, loc, loc_city, query_key, query_loc, ftype, plist] = his.split('\x03')

            query_loc = query_loc.strip()
            loc_city = loc_city.split(' ')[0]
            if len(query_key) < 2 or query_key in stop_querys:
                failed_query_count += 1
                continue
            query_key = clean_key(query_key, split_characters)
            if len(loc_city) < 2:
                loc_city = 'unknown'
            time_stamp = parse_datetime(dtime)

            query = {
                'time_stamp': time_stamp,
                'dtime': dtime,
                'loc_city': loc_city,
                'query_key': query_key,
                'query_loc': query_loc,
                'ftype': ftype,
                'plist': plist
            }
            query_list = map_query_data.get(cuid, [])
            if len(query_list) < 1 or time_stamp >= query_list[-1]['time_stamp']:
                query_list.append(query)
            else:
                pos = find_appropriate_position(query_list, query)
                query_list.insert(pos, query)
            map_query_data[cuid] = query_list

        line = raw_data_file.readline()

    print 'query number: %f, failed number %f' % (query_history_count, failed_query_count)
    print 'failed rate: %f' % (failed_query_count/query_history_count)

    # check time order
    pass_flag = True
    for cuid, query_list in map_query_data.iteritems():
        for i in xrange(len(query_list)-1):
            if query_list[i]['time_stamp'] > query_list[i+1]['time_stamp']:
                print 'time order error at cuid: %s' % cuid
                pass_flag = False
    if pass_flag:
        print 'time order check pass!'
    else:
        print 'time order check not pass!'



    data_set = {'map_query_data': map_query_data}
    return data_set

def gen_service_dataset(params, map_query_data, poi_data):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='data_set_name', default='beijing_12_highfreq3')
    parser.add_argument('-p', '--poidata', dest='poi_data_path', default='beijing_poi_data.pkl')
    parser.add_argument('--get_service', dest='generate_service', type=int, default=0)
    # parser.add_argument('--get_voca', dest='get_voca', type=int, default=0)
    args = parser.parse_args()

    params = vars(args)
    params['data_set_name'] = params['data_set_name'].lower()

    params['proj_folder'] = get_proj_folder()

    print json.dumps(params)

    map_query_data_set = gen_data_set(params)
    data_file_path = os.path.join(params['data_folder'], 'raw_data_set.pkl')
    data_file = open(data_file_path, 'wb')
    pickle.dump(map_query_data_set, data_file)
    data_file.close()
    print 'save data into %s' % data_file_path

    if params['generate_service']:
        poi_data = pickle.load(open(params['poidata'], 'rb'))
        gen_service_dataset(params, map_query_data_set, poi_data)
