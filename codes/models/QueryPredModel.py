__author__ = 'SongJun-Dell'
import random
import numpy as np
from ml_idiot.nn.models.BasicModel import BasicModel, get_data_splits

def get_model_batch_test_result(model, batch_data):
    res = model.test_on_batch(batch_data)
    return res

def get_train_cost_one_batch(model, batch_data, mode):
    loss_cost, grad_params = model.get_raw_loss(batch_data, mode=mode)
    return {'loss_cost': loss_cost, 'grad_params': grad_params}

def get_batch_predict_candidates(model, batch_data):
    res = model.get_predict_candidates(batch_data)
    # print 'get batch predict candidates.'
    return res

class QueryPredModel(BasicModel):
    def __init__(self, state):
        super(QueryPredModel, self).__init__(state)

    def get_cost(self, batch_data, pool=None, num_process=0, mode='test'):
        cost, grad_params = self.train_one_batch(batch_data, pool, num_process, mode)
        if mode == 'train':
            return cost, grad_params
        elif mode == 'test':
            return cost
        else:
            raise StandardError('mode error')

    def train_one_batch(self, batch_data, pool=None, num_processes=0, mode='train'):
        if pool is None:
            train_res = get_train_cost_one_batch(self, batch_data, mode)
            train_cost = train_res['loss_cost']
            grad_params = train_res['grad_params']
        else:
            batch_splits = get_data_splits(num_processes, batch_data)
            train_cost = 0
            grad_params = None
            pool_results = list()
            for pro in xrange(num_processes):
                data_samples = batch_splits[pro]
                random.shuffle(data_samples)
                tmp_result = pool.apply_async(get_train_cost_one_batch, (self, data_samples, mode))
                pool_results.append(tmp_result)

            for p in xrange(num_processes):
                split_res = pool_results[p].get()
                train_cost += split_res['loss_cost']
                grad_params = self.merge_grads(grad_params, split_res['grad_params'], scale=float(num_processes))

            train_cost /= num_processes

        self.grad_regularization(grad_params)
        # print 'train_cost %f' % train_cost
        train_cost = train_cost + self.state['regularize_rate'] * self.get_regularization()
        return train_cost, grad_params

    def get_raw_loss(self, batch_data, mode='test'):
        batch_query_preds, train_batch_cache = self.forward_batch(batch_data, mode='train')
        batch_cost = self.get_batch_cost(batch_query_preds)
        if mode == 'train':
            grad_params = self.calculate_grads(train_batch_cache)
            return batch_cost, grad_params
        elif mode == 'test':
            return batch_cost, None
        else:
            raise StandardError('mode error')

    def calculate_grads(self, batch_cache_data):
        grad_params = self.init_grads()
        self.backward_batch(grad_params, batch_cache_data)
        return grad_params

    def backward_batch(self, grad_params, batch_cache_data):
        """
        :param batch_cache_data: [ sample_cache_data, ... ]
        sample_cache_data = [ query_cache_data, ... ]
        query_cache_data = {
            'gth_query_idxs': [query_key_idx, ...],
            'gth_cuid_idxs': [cuid_idx, ...],
            'gth_city_idxs': [city_idx, ...],
            'pred_probs': (query_size, vocab_size)
            'session_lstm_cache': lstm_cache,
            'encode_lstm_caches':[ lstm_cache, ... ],
            'decode_lstm_caches':[ lstm_cache, ... ]
            'out_mapping_in_vecs': (query_size, out_mapping_in_dim)
        }
        'lstm_cache': {
            'cell': cell,
            'lstm_in_vec': lstm_in_vec,
            'prev_cell': prev_cell,
            'input_gate': input_gate,
            'forget_gate': forget_gate,
            'out_gate': out_gate,
            'hidden_vec': hidden_vec,
        }
        :return:
        """
        self.temp_values['batch_size'] = len(batch_cache_data)
        for sample_cache_data in batch_cache_data:
            self.backward_sample(grad_params, sample_cache_data)

    def forward_batch(self, batch_data, mode='train'):
        """
        :param batch_data: [ sample, ... ]
            sample = {
                'cuid_idx': cuid_idx,
                'indexed_sesssion_queries': [ session_queries, ... ]
            }
            indexed_se_queries = {
                'loc_city_idxs': [ city_idx, ... ]
                'query_key_idxs': [ query_key_idx, ... ]
            }
        :return: batch_query_preds = [ sample_preds, ... ]
            sample_preds = [ query_preds, ... ]
            query_preds = {
                'gth_idx': [ query_key_idx, ... ]
                'pred_probs': ( query_size, vocab_size )
            }
        """
        batch_query_preds = list()
        batch_cache = list()
        for sample in batch_data:
            query_preds, sample_cache = self.forward_sample(sample, mode)
            if mode == 'train':
                batch_cache.append(sample_cache)
            batch_query_preds.append(query_preds)
        return batch_query_preds, batch_cache

    def get_batch_cost(self, batch_query_preds):
        """
        :param batch_query_preds = [ sample_preds, ... ]
            sample_preds = [ query_preds, ... ]
            query_preds = {
                'gth_idx': [ query_key_idx, ... ]
                'pred_probs': ( query_size, vocab_size )
            }
        :return:
        """
        batch_cost = list()
        for sample_preds in batch_query_preds:
            sample_cost = self.get_sample_cost(sample_preds)
            # print sample_cost
            batch_cost.append(sample_cost)
        return np.mean(batch_cost)

    def get_pred_candidates(self, batch_query_preds):
        batch_size = len(batch_query_preds)
        pred_candidates = list()
        for si in xrange(batch_size):
            sample_preds = batch_query_preds[si]
            query_candis = list()
            for qi in xrange(len(sample_preds)):
                pred_probs = sample_preds[qi]['pred_probs']
                sorted_idxs = np.argsort(pred_probs)[:,::-1]
                candis = sorted_idxs.transpose()[0]
                query_candis.append(list(candis))
            pred_candidates.append(query_candis)

        return pred_candidates

    def get_batch_ranks(self, batch_query_preds):
        batch_size = len(batch_query_preds)
        ranks = list()
        for si in xrange(batch_size):
            # print 'num query: %d' % len(batch_query_preds)
            sample_preds = batch_query_preds[si]
            for qi in xrange(len(sample_preds)):
                query_preds = sample_preds[qi]
                gth_idxs = query_preds['gth_idx']
                pred_probs = query_preds['pred_probs']
                for ki,idx in enumerate(gth_idxs):
                    preds = pred_probs[ki,:]
                    rank = self.get_rank(preds, idx)
                    ranks.append(rank)
        return ranks

    def get_rank(self, probabilities, idx):
        sorted_idx = np.argsort(probabilities)[::-1]
        rank = np.where(sorted_idx==idx)[0][0] + 1
        return rank

    def test_on_batch(self, batch_data):
        batch_query_preds, _ = self.forward_batch(batch_data, mode='test')
        batch_cost = self.get_batch_cost(batch_query_preds)

        ranks = self.get_batch_ranks(batch_query_preds)
        return {'cost':batch_cost, 'ranks':ranks}

    def get_predict_candidates(self, batch_data):
        batch_query_preds, _ = self.forward_batch(batch_data, mode='test')
        batch_cost = self.get_batch_cost(batch_query_preds)
        ranks = self.get_batch_ranks(batch_query_preds)
        candidates = self.get_pred_candidates(batch_query_preds)
        return {'cost': batch_cost, 'ranks':ranks, 'predict_candidates':candidates}

    def test(self, batch_data, pool=None, num_processes=0):
        self.set_unknow_embedding()
        if pool is None:
            res = get_model_batch_test_result(self, batch_data)
            res['cost'] += self.state['regularize_rate'] * self.get_regularization()
            return res
        else:
            cost = 0
            ranks = list()
            batch_splits = get_data_splits(num_processes, batch_data)
            num_split = len(batch_splits)
            pool_results = list()
            for pro in xrange(num_split):
                data_samples = batch_splits[pro]
                random.shuffle(data_samples)
                tmp_result = pool.apply_async(get_model_batch_test_result, (self, data_samples))
                pool_results.append(tmp_result)

            for p in xrange(num_split):
                split_res = pool_results[p].get()
                cost += split_res['cost']
                ranks += split_res['ranks']

            cost /= num_split
            cost += self.state['regularize_rate'] * self.get_regularization()
            return {'cost': cost, 'ranks':ranks}

    def predict_candidates(self, batch_data, pool=None, num_processes=0):
        self.set_unknow_embedding()
        if pool is None:
            res = get_batch_predict_candidates(self, batch_data)
            return res
        else:
            cost = 0
            ranks = list()
            batch_candidates = list()
            batch_splits = get_data_splits(num_processes, batch_data)
            pool_results = list()
            for pro in xrange(num_processes):
                data_samples = batch_splits[pro]
                temp_result = pool.apply_async(get_batch_predict_candidates, (self, data_samples))
                pool_results.append(temp_result)

            for p in xrange(num_processes):
                split_res = pool_results[p].get()
                batch_candidates += split_res['predict_candidates']
                cost += split_res['cost']
                ranks += split_res['ranks']
            cost /= num_processes
            cost += self.state['regularize_rate'] * self.get_regularization()

            return {'predict_candidates': batch_candidates, 'cost': cost, 'ranks': ranks}
