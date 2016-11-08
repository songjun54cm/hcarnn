__author__ = 'SongJun-Dell'
import argparse
import datetime
import json
import random
import cPickle as pickle
import os
import numpy as np
from ml_idiot.data_provider.BasicDataProvider import BasicDataProvider
from utils import sort_dict, get_voca_from_count, get_proj_folder, get_data_set, counting_time
def get_prob(freq_list):
    probs = np.array(freq_list, dtype=np.float32)
    probs /= np.sum(probs)
    probs += 1e-20
    probs = np.log(probs)
    probs -= np.max(probs)
    return probs

class SessDataProvider(BasicDataProvider):
    def __init__(self):
        super(SessDataProvider, self).__init__()
        self.split_running_batches = {
            'train': list(),
            'valid': list(),
            'test': list(),
            'train_valid': list()
        }
        self.data_set_params = dict()
        self.cuid_voca = None
        self.query_voca = None
        self.city_voca = None

    @counting_time
    def build(self, data_set, params=None):
        map_query_data = data_set['map_query_data']
        if params is None:
            chunk_len = {'days':2} # days
            session_len = {'days':1}
            train_rate = 0.8
            valid_rate = 0.1
            test_rate = 0.1
        else:
            chunk_len = params.get('query_chunk_len', {'days':2})
            session_len = params.get('query_session_len', {'days':1})
            train_rate = params.get('train_rate', 0.8)
            valid_rate = params.get('valid_rate', 0.1)
            test_rate = params.get('test_rate', 0.1)

        seq_samples = self.get_chunk_samples(map_query_data, session_len, chunk_len)

        splits_datas = self.get_splits_data(seq_samples, [train_rate, valid_rate, test_rate])
        train_data = splits_datas[0]
        valid_data = splits_datas[1]
        test_data = splits_datas[2]

        vocabs = self.get_voca_from_train_data(train_data)

        self.splits = {
            'train': self.reorder_data(self.make_indexed_data(train_data, vocabs)),
            'valid': self.reorder_data(self.make_indexed_data(valid_data, vocabs)),
            'test': self.reorder_data(self.make_indexed_data(test_data, vocabs))
        }

        self.splits['train_valid'] = self.reorder_data(random.sample(self.splits['train'], len(self.splits['valid'])))

        self.post_vocabs(vocabs)
        self.print_statistics()
        print 'build data provider finish.'

    def get_splits_data(self, samples_list, rate_list):
        num_sample = len(samples_list)
        random_idx = range(0, num_sample)
        random.seed(0)
        random.shuffle(random_idx)
        split_lens = [ int(srate * num_sample) for srate in rate_list]
        split_idxes = list()
        split_data_list = list()
        spos = 0
        epos = 0
        for pos in split_lens:
            epos += pos
            split_idxes.append(random_idx[spos:epos])
            spos = epos

        for idxes in split_idxes:
            split_data_list.append([samples_list[di] for di in idxes])
        return split_data_list

    def post_vocabs(self, vocabs):
        self.cuid_voca = vocabs['cuid_voca']
        self.cuid_voca['cuid_prob'] = get_prob(vocabs['cuid_voca']['cuid_freq'])

        self.city_voca = vocabs['city_voca']
        self.city_voca['city_prob'] = get_prob(vocabs['city_voca']['city_freq'])

        self.query_voca = vocabs['query_voca']
        self.query_voca['query_prob'] = get_prob(vocabs['query_voca']['query_freq'])

        self.data_set_params['cuid_num'] = len(self.cuid_voca['ix_to_cuid'])
        self.data_set_params['city_num'] = len(self.city_voca['ix_to_city'])
        self.data_set_params['query_num'] = len(self.query_voca['ix_to_query'])

    def print_statistics(self):
        print 'num train: %d' % len(self.splits['train'])
        print 'num valid: %d' % len(self.splits['valid'])
        print 'num test: %d' % len(self.splits['test'])
        print 'num query %d' % self.data_set_params['query_num']

    def get_chunk_samples(self, map_query_data, session_len, chunk_len):
        """

        :param map_query_data:
        :param session_len:
        :param chunk_len:
        :return: chunk_samples = [ query_chunk, ... ]
        query_chunk = {
            'cuid': cuid,
            'session_queries': [ query_session, ...  ]
        }
        query_session = [ query, ... ]
        """
        session_time_interval = datetime.timedelta(**session_len)
        chunk_time_interval = datetime.timedelta(**chunk_len)
        chunk_samples = list()
        for cuid in map_query_data:
            cuid_query_list = map_query_data[cuid]
            session_queries = list()
            cur_date = cuid_query_list[0]['time_stamp']
            cur_day = datetime.datetime(cur_date.year, cur_date.month, cur_date.day)

            session_start_time = cur_date
            session_end_time = session_start_time + session_time_interval
            chunk_start_time = cur_day
            chunk_end_time = chunk_start_time + chunk_time_interval

            query_chunk = {'cuid':cuid, 'session_queries':[]}
            chunk_sessions = []
            query_session = []
            for query in cuid_query_list:
                if query['time_stamp'] >= chunk_end_time:
                    chunk_sessions.append(query_session)
                    query_chunk['session_queries'] = chunk_sessions
                    chunk_samples.append(query_chunk)

                    cur_date = query['time_stamp']
                    cur_day = datetime.datetime(cur_date.year, cur_date.month, cur_date.day)
                    chunk_start_time = cur_day
                    chunk_end_time = chunk_start_time + chunk_time_interval

                    session_start_time = cur_date
                    session_end_time = session_start_time + session_time_interval

                    chunk_sessions = []
                    query_chunk = {'cuid':cuid, 'session_queries':[]}
                    query_session = []
                elif query['time_stamp'] > session_end_time:
                    chunk_sessions.append(query_session)

                    session_start_time = query['time_stamp']
                    session_end_time = session_start_time + session_time_interval
                    query_session = []
                query_session.append(query)

            if len(query_session)>0:
                chunk_sessions.append(query_session)
            if len(chunk_sessions)>0:
                query_chunk['session_queries'] = chunk_sessions
                chunk_samples.append(query_chunk)
        return chunk_samples

    def get_session_day_samples(self, map_query_data):
        """
        get session day samples
        :param map_query_data:
        :return: session_day_samples = {
            cuid: session_day_queries
        }
            session_day_queries = [ day_queries, ...]
            day_queries = {
                'day_time': day time,
                'queries': [ query, ... ]
                }
        """
        session_day_samples = dict()
        for cuid in map_query_data:
            query_list = map_query_data[cuid]
            session_day_queries = []
            cur_date = query_list[0]['time_stamp']
            cur_day = cur_date.day
            day_queries = { 'day_time': datetime.datetime(cur_date.year, cur_date.month, cur_date.day),
                            'queries': []}
            for query in query_list:
                if cur_day < query['time_stamp'].day:
                    session_day_queries.append(day_queries)
                    cur_date = query['time_stamp']
                    cur_day = cur_date.day
                    day_queries = { 'day_time': datetime.datetime(cur_date.year, cur_date.month, cur_date.day),
                                    'queries': []}

                day_queries['queries'].append(query)
            if len(day_queries['queries']) > 0:
                session_day_queries.append(day_queries)
            session_day_samples[cuid] = session_day_queries
        return session_day_samples

    def get_sequence_samples(self, session_day_samples, wind_len):
        """
        :param session_day_samples: { cuid: session_day_queries }
            session_day_queries = [ day_queries, ... ]
            day_queries = {
                'day_time': day time,
                'queries': [ query, ... ]
                }
        :param wind_len:
        :return: seq_samples = [ seq_sample, ... ]
            seq_sample = {
                'cuid': cuid,
                'session_queries': [ se_queries, ... ]
            }
            se_queries = [ query, ... ]
        """

        time_interval = datetime.timedelta(**wind_len)
        seq_samples = list()
        for cuid in session_day_samples:
            session_day_queries = session_day_samples[cuid]
            seq_sample = {
                'cuid': cuid,
                'session_queries': []
            }
            start_day_time = session_day_queries[0]['day_time']
            end_day_time = start_day_time + time_interval
            for day_queries in session_day_queries:
                if day_queries['day_time'] >= end_day_time:
                    seq_samples.append(seq_sample)
                    start_day_time = day_queries['day_time']
                    end_day_time = start_day_time + time_interval
                    seq_sample = {
                        'cuid': cuid,
                        'session_queries': []
                    }
                seq_sample['session_queries'].append(day_queries['queries'])
            if len(seq_sample['session_queries']) > 0:
                seq_samples.append(seq_sample)
        return seq_samples

    def get_voca_from_train_data(self, train_data):
        """
        :param train_data: [ seq_sample, ... ]
        :return:
        """
        cuid_count = dict()
        city_count = dict()
        query_count = dict()
        for seq_sample in train_data:
            cuid = seq_sample['cuid']
            cuid_count[cuid] = cuid_count.get(cuid, 0) + 1
            session_queries = seq_sample['session_queries']
            for se_queries in session_queries:
                for query in se_queries:
                    loc_city = query['loc_city']
                    city_count[loc_city] = city_count.get(loc_city, 0) + 1
                    query_key = query['query_key']
                    query_count[query_key] = query_count.get(query_key, 0) + 1
        cuid_count = sort_dict(cuid_count)
        city_count = sort_dict(city_count)
        query_count = sort_dict(query_count)

        tmp_voca = get_voca_from_count(cuid_count)
        cuid_voca = {
            'cuid_to_ix': tmp_voca['w_to_ix'],
            'ix_to_cuid': tmp_voca['ix_to_w'],
            'cuid_freq': tmp_voca['w_count']
        }
        tmp_voca = get_voca_from_count(city_count)
        city_voca = {
            'city_to_ix': tmp_voca['w_to_ix'],
            'ix_to_city': tmp_voca['ix_to_w'],
            'city_freq': tmp_voca['w_count']
        }
        tmp_voca = get_voca_from_count(query_count, ['<START>'])
        query_voca = {
            'query_to_ix': tmp_voca['w_to_ix'],
            'ix_to_query': tmp_voca['ix_to_w'],
            'query_freq': tmp_voca['w_count']
        }
        return {'cuid_voca': cuid_voca, 'city_voca': city_voca, 'query_voca': query_voca}

    def shuffle_data(self, datas):
        random.shuffle(datas)
        return datas

    def reorder_data(self, datas):
        sample_lengths = [self.get_sample_length(samp) for samp in datas]
        ordered_idx = np.argsort(sample_lengths)[::-1]
        ordered_data = [datas[i] for i in ordered_idx]
        return ordered_data

    def get_sample_length(self, sample):
        sample_length = 0
        for queries in sample['indexed_session_queries']:
            sample_length += len(queries['query_key_idxs'])
        return sample_length

    def make_indexed_data(self, datas, vocabs):
        """
        :param datas: [seq_sample, ... ]
            seq_sample = {
                'cuid': cuid,
                'session_queries': [ se_queries, ... ]
            }
            se_queries = [ query, ... ]
            query = {
                'time_stamp': time stamp,
                'dtime': date time,
                'loc_city': locate city,
                'query_key': query key,
                'query_loc': query locate city
                'ftype': function type,
                'plist': poi list
            }
        :param vocabs:{
            'cuid_voca': { 'cuid_to_ix': dict, 'ix_to_cuid': list, 'cuid_freq': list },
            'city_voca':
            'query_voca':
            }
        :return: indexed_data: [ indexed_sample, ... ]
            indexed_sample = {
                'cuid_idx': cuid_idx,
                'indexed_session_queries': [ indexed_se_queries, ... ]
            }
            indexed_se_queries = {
                'loc_city_idxs': [ city_idx, ... ],
                'query_key_idxs': [ query_key_idx, ... ]
            }
        """
        cuid_to_ix = vocabs['cuid_voca']['cuid_to_ix']
        city_to_ix = vocabs['city_voca']['city_to_ix']
        query_to_ix = vocabs['query_voca']['query_to_ix']
        indexed_data = list()
        for seq_sample in datas:
            cuid = seq_sample['cuid']
            session_queries = seq_sample['session_queries']
            cuid_idx = cuid_to_ix.get(cuid, 0)
            indexed_sample = {
                'cuid_idx': cuid_idx,
                'indexed_session_queries': []
            }
            for se_queries in session_queries:
                loc_city_idxs = []
                query_key_idxs = []
                # query_times = []
                for query in se_queries:
                    loc_city_idxs.append(city_to_ix.get(query['loc_city'], 0))
                    query_key_idxs.append(query_to_ix.get(query['query_key'], 0))
                    # query_times.append(query['time_stamp'])
                indexed_se_queries = {
                    'loc_city_idxs': loc_city_idxs,
                    'query_key_idxs': query_key_idxs
                }
                indexed_sample['indexed_session_queries'].append(indexed_se_queries)
            indexed_data.append(indexed_sample)

        return indexed_data

    def cooccur_iter_splits_samples(self, splits):
        datas = self.splits[splits]
        for data in datas:
            cuid = data['cuid_idx']
            for query in data['indexed_session_queries']:
                city_idxs = query['loc_city_idxs']
                query_idxs = query['query_key_idxs']
                yield (cuid, city_idxs[0], city_idxs, query_idxs)

    def get_markov_splits(self, splits):
        datas = self.splits[splits]
        markov_datas = list()
        for data in datas:
            cuid = data['cuid_idx']
            for query in data['indexed_session_queries']:
                city_idxs = query['loc_city_idxs']
                query_idxs = query['query_key_idxs']
                markov_datas.append((cuid, city_idxs[0], city_idxs, query_idxs))
        return markov_datas

    def transform_to_flat_lstm(self):
        for splits in self.splits.keys():
            temp_data_splits = list()
            for data in self.splits[splits]:
                temp_data_splits += self.get_flat_lstm_datas(data)
            self.splits[splits] = temp_data_splits

    def get_flat_lstm_datas(self, data):
        tmp_datas = list()
        cuid = data['cuid_idx']
        for query in data['indexed_session_queries']:
            city_idxs = query['loc_city_idxs']
            query_idxs = query['query_key_idxs']
            tmp_datas.append((cuid, city_idxs[0], city_idxs, query_idxs))
        return tmp_datas

    def transform(self, data):
        return data

    def iter_splits_batches(self, batch_size, split):
        batch = list()
        datas = self.get_split(split)
        for d in datas:
            batch.append(d)
            if len(batch) >= batch_size:
                yield batch
                batch = list()

        if len(batch) > 0:
            yield batch

    def iter_split_running_batches(self, orig_batch_size, split, mode='static'):
        batch_split = self.split_running_batches.get(split, [])
        if len(batch_split) == 0:
            self.split_running_batches[split] = self.get_split_running_batches(orig_batch_size, split, mode)
        for batch in self.split_running_batches[split]:
            yield batch
    def get_split_running_batches(self, orig_batch_size, split, mode='static'):
        running_batches = list()
        cur_batch_size = orig_batch_size
        batch = list()
        datas = self.get_split(split)
        if mode == 'static':
            for d in datas:
                batch.append(d)
                if len(batch) >= cur_batch_size:
                    running_batches.append(batch)
                    batch = list()
            if len(batch) > 0:
                running_batches.append(batch)

        elif mode == 'dynamic':
            cur_batch_size = 20
            cur_sample_len = self.get_sample_length(datas[0])
            for d in datas:
                batch.append(d)
                if len(batch) >= cur_batch_size:
                    min_sample_len = self.get_sample_length(batch[-1])
                    running_batches.append(batch)
                    batch = list()
                    rate = max(int(cur_sample_len/min_sample_len), 1)
                    cur_sample_len = int(cur_sample_len / rate)
                    cur_batch_size = min(cur_batch_size * rate, orig_batch_size)
            if len(batch) > 0:
                running_batches.append(batch)
        print 'finish create running batches.'
        return running_batches
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='data_set_name',
                        default='toy_beijing_session30minute_chunk3day',
                        help='data_set_name: toy/'
                         'toy_beijing/'
                         'toy_beijing_session30minute_chunk3day'
                         'beijing_12_highfreq3/'
                         'beijing_12_highfreq3_win3/'
                         'beijing_12_highfreq3_win4/'
                         'beijing_12_highfreq3_win5/'
                         'beijing_12_highfreq3_win1/'
                         'beijing_12_highfreq_win30minute/'
                         'beijing_12_highfreq3_session30minute_chunk3day/'
                         'beijing_12_highfreq3_session1hour_chunk3day')
    args = parser.parse_args()

    params = vars(args)
    return params

if __name__ == '__main__':
    random.seed(12345)
    params = get_params()
    params['data_set_name'] = params['data_set_name'].lower()
    params['proj_folder'] = get_proj_folder()
    params['data_folder'] = os.path.join(params['proj_folder'], 'data', params['data_set_name'])
    if not os.path.exists(params['data_folder']):
        os.mkdir(params['data_folder'])
        print 'make new dir %s' % params['data_folder']

    print json.dumps(params,indent=2)

    data_provider = SessDataProvider()

    dp_file_path = os.path.join(params['data_folder'], 'session_data_provider.pkl')
    if os.path.exists(dp_file_path):
        print 'data provider already exists in %s' % params['data_folder']
    else:
        data_set = get_data_set(params)
        data_provider.build(data_set, params)
        data_provider.save(dp_file_path)
        print 'build data provider and save to %s' % dp_file_path