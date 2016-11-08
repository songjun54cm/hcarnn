import random
__author__ = 'SongJun-Dell'
import os, json, time, sys
import numpy as np
import cPickle as pickle
from multiprocessing import Pool
from ml_idiot.solver.NeuralNetworkSolver import NeuralNetworkSolver

def get_mrr(ranks):
    rs = np.array(ranks, dtype=np.float)
    mrr = np.mean( 1.0 / rs )
    mean_rank = np.mean(rs)
    return mrr, mean_rank

def get_model_cost(model, batch_data):
    loss_cost, grad_params = model.get_cost(batch_data, mode='train')
    return {'loss_cost': loss_cost, 'grad_params': grad_params}

class ParallelNormalSolver(NeuralNetworkSolver):
    def __init__(self, state):
        super(ParallelNormalSolver, self).__init__(state)
        self.state = state
        # self.log_file = open(os.path.join(state['out_folder'], 'log.log'), 'wb')
        self.last_loss = np.inf
        self.top_valid_cost = np.inf
        self.top_valid_mrr = 0
        self.top_valid_mrank = np.inf
        self.smooth_valid_loss = dict()

        self.batch_size = state['batch_size']
        self.max_epoch = state['max_epoch']
        self.valid_epoch = state['valid_epoch']
        self.valid_batch_size = state['valid_batch_size']
        self.cost_scale = state['cost_scale']
        self.pool = None
        self.grad_cache = dict()
        if state['parallel']:
            print 'create parallel pool with %d processes.' % self.state['num_processes']
            self.pool = Pool(processes=self.state['num_processes'])

    def train_one_batch(self, model, batch_data, epoch_i):
        self.iter_count += 1
        t0 = time.time()
        # print 'batch size: %d' % len(batch_data)
        if self.state['parallel']:
            # loss_cost = self.parallel_train_process_one_batch(model, batch_data)
            loss_cost, grad_params = model.train_one_batch(batch_data, self.pool, self.state['num_processes'])
        else:
            loss_cost, grad_params = model.train_one_batch(batch_data)
            # loss_cost = self.train_process_one_batch(model, batch_data)
        if self.iter_count == 0:
            model.print_params()
            print grad_params

        self.update_model(model, grad_params)

        loss_cost *= self.cost_scale
        self.valid_sample_count += len(batch_data)
        # calculate smooth cost
        if self.iter_count == 1:
            self.smooth_train_cost = loss_cost
        else:
            self.smooth_train_cost = 0.99 * self.smooth_train_cost + 0.01 * loss_cost
        # print message
        time_eclipse = time.time() - t0
        self.sample_count += len(batch_data)

        epoch_rate = epoch_i + 1.0 * self.sample_count / self.train_size
        message = 'samples %d/%d done in %.3fs. epoch %.3f/%d. loss_cost= %f, (smooth %f)' \
            % (self.sample_count, self.train_size, time_eclipse, epoch_rate, self.max_epoch, loss_cost, self.smooth_train_cost)
        self.log_message(message)
        # detect loss exploding
        if not self.detect_loss_explosion(loss_cost):
            sys.exit()

    def to_validate(self, epoch_i):
        return (self.valid_sample_count >= self.valid_sample_num) or \
               (epoch_i>=self.max_epoch-1 and self.sample_count>=self.train_size)

    def train(self, model, data_provider, state):
        self.create_checkpoint_dir()
        # log state
        self.log_message(json.dumps(state, indent=2))
        self.setup_train_state(data_provider)

        for epoch_i in xrange(self.max_epoch):
            self.sample_count = 0
            self.epoch_i = epoch_i
            for batch_data in data_provider.iter_split_running_batches(self.batch_size, split='train'):
                self.train_one_batch(model, batch_data, epoch_i)

                # validation
                if self.to_validate(epoch_i):
                    self.valid_sample_count = 0
                    self.valid_count += 1
                    # validate on the train valid data set
                    res = self.validate_on_split(model, data_provider, split='train_valid')

                    # validate on the validate data set
                    res = self.validate_on_split(model, data_provider, split='valid')

                    # validate on the test data set
                    res = self.validate_on_split(model, data_provider, split='test')

                    self.detect_to_save(res, model)

        return self.last_save_model_file_path

    def validate_on_split(self, model, data_provider, split):
        t0 = time.time()
        res = self.test_on_split(model, data_provider, split)
        valid_num = res['sample_num']
        valid_loss_cost = res['loss_cost']
        valid_ranks = res['ranks']
        mrr, mean_rank = get_mrr(valid_ranks)
        self.get_rks(valid_ranks, [1,5,10,15,20])
        if self.valid_count == 1:
            self.smooth_valid_loss[split] = valid_loss_cost
        else:
            self.smooth_valid_loss[split] = 0.99 * self.smooth_valid_loss[split] + 0.01 * valid_loss_cost
        time_eclipse = time.time() - t0
        message = 'evaluate %d %s samples in %.3fs and get cost %f (smooth %f) and mrr: %f, mean_rank: %f.'\
                    % (valid_num, split, time_eclipse, valid_loss_cost, self.smooth_valid_loss[split], mrr, mean_rank)
        self.log_message(message)
        results = {
            'mrr': mrr,
            'mean_rank': mean_rank,
            'loss': valid_loss_cost
        }
        return results

    def test(self, model, data_provider):
        t0 = time.time()
        res = self.test_on_split(model, data_provider, 'test')
        mrr, mean_rank = get_mrr(res['ranks'])
        time_eclipse = time.time() - t0
        message = 'test in %.3fs. loss_cost= %f, mrr=%f, mean_rank=%f' \
            % (time_eclipse, res['loss_cost'], mrr, mean_rank)
        self.log_message(message)

    def test_on_split(self, model, data_provider, split):
        sample_num = 0
        batch_num = 0
        ranks = []
        loss_cost = 0
        for batch_data in data_provider.iter_split_running_batches(self.valid_batch_size, split, 'static'):
            sample_num += len(batch_data)
            batch_num += 1
            test_res = model.test(batch_data, self.pool, self.state['num_processes'])
            batch_loss = test_res['cost']
            batch_ranks = test_res['ranks']
            loss_cost += batch_loss
            ranks += batch_ranks
        loss_cost /= batch_num
        loss_cost *= self.cost_scale
        return {'sample_num': sample_num,
                'batch_num': batch_num,
                'ranks': ranks,
                'loss_cost': loss_cost
                }

    def process_before_test(self, data_batch):
        if self.state['model_name'] == 'normal_flat_lstm':
            tmp_datas = list()
            for data in data_batch:
                cuid = data['cuid_idx']
                for query in data['indexed_session_queries']:
                    city_idxs = query['loc_city_idxs']
                    query_idxs = query['query_key_idxs']
                    tmp_datas.append((cuid, city_idxs[0], city_idxs, query_idxs))
            return tmp_datas
        else:
            return data_batch

    def test_on_batch(self, model, idata_batch):
        data_batch = self.process_before_test(idata_batch)
        t0 = time.time()
        sample_num = len(data_batch)
        ranks = list()
        batch_costs = list()
        start_pos = 0
        while start_pos <= sample_num:
            end_pos = start_pos + self.valid_batch_size
            mini_batch = data_batch[start_pos:end_pos]
            test_res = model.test(mini_batch, self.pool, self.state['num_processes'])
            batch_costs.append(test_res['cost'])
            batch_ranks = test_res['ranks']
            ranks += batch_ranks
            start_pos = end_pos
        loss_cost = self.cost_scale * sum(batch_costs)/len(batch_costs)
        mrr, mean_rank = get_mrr(ranks)
        self.get_rks(ranks, [1,5,10,15,20])
        time_eclipse = time.time() - t0
        message = 'test in %.3fs. loss_cost= %f, mrr=%f, mean_rank=%f' \
            % (time_eclipse, loss_cost, mrr, mean_rank)
        print message

    def predict_candidates_batch(self, model, batch_data):
        start_pos = 0
        num_data = len(batch_data)
        res = {'predict_candidates': list(), 'cost': 0.0, 'ranks': list()}
        while start_pos <= num_data:
            min_batch = batch_data[start_pos: start_pos+self.state['valid_batch_size']]
            min_res = model.predict_candidates(min_batch, self.pool, self.state['num_processes'])

            res['predict_candidates'] += min_res['predict_candidates']
            res['cost'] += min_res['cost']
            res['ranks'] += min_res['ranks']
            start_pos += self.state['valid_batch_size']
            print '%d/%d sample predicted.' % (start_pos, num_data)
        return res

    def predict_candidates(self, model, data_provider, split):
        t0 = time.time()
        all_candidates = list()
        cost = 0
        ranks = list()
        batch_num = 0
        for batch_data in data_provider.iter_split_running_batches(self.valid_batch_size, split, 'static'):
            batch_num += 1
            print 'batch %d predicted.' % batch_num
            res = self.predict_candidates_batch(model, batch_data)
            all_candidates += res['predict_candidates']
            cost += res['cost']
            ranks += res['ranks']
        cost /= batch_num
        eclipse_time = time.time() - t0
        print 'predict finished in %.3f seconds' % eclipse_time
        return {'batch_num': batch_num,
                'cost': cost,
                'ranks': ranks,
                'predict_candidates': all_candidates
                }

    def detect_loss_explosion(self, loss):
        if loss > self.smooth_train_cost * 10:
            message = 'Aborting, loss seems to exploding. try to run gradient check or lower the learning rate.'
            self.log_message(message)
            return False
        # self.smooth_train_cost = loss
        return True

    def get_batch_splits(self, num_patch, batch_data):
        batch_splits = list()
        batch_size = len(batch_data)
        step = int(batch_size/num_patch)
        idxs = range(0, batch_size, step)
        if batch_size % step == 0:
            idxs.append(batch_size)
        else:
            idxs[-1] = batch_size
        for i in xrange(num_patch):
            batch_splits.append(batch_data[idxs[i]:idxs[i+1]])
        return batch_splits

    def get_forward_attention_caches(self, model, batch_data):
        t0 = time.time()
        res = self.get_caches(model, batch_data)
        data_num = res['sample_num']
        loss = res['loss']
        ranks = res['ranks']
        mrr, mean_rank = get_mrr(ranks)
        self.get_rks(ranks, [1,5,10,15,20])
        time_eclipse = time.time() - t0
        message = 'evaluate %d samples in %.3fs and get cost %f and mrr: %f, mean_rank: %f.'\
                    % (data_num, time_eclipse, loss, mrr, mean_rank)
        print message
        results = {
            'mrr': mrr,
            'mean_rank': mean_rank,
            'loss': loss,
            'attention_caches': res['attend_caches']
        }
        return results

    def get_caches(self, model, batch_data):
        sample_num = len(batch_data)
        res = model.get_cache(batch_data, self.pool, self.state['num_processes'])
        res['sample_num'] = sample_num
        return res

    def transform_data(self, model_name, batch_data):
        if model_name == 'normal_flat_lstm':
            datas = self.get_flat_lstm_datas(batch_data)
        else:
            datas = batch_data
        return datas

    def get_flat_lstm_datas(self, datas):
        tmp_datas = list()
        for data in datas:
            cuid = data['cuid_idx']
            for query in data['indexed_session_queries']:
                city_idxs = query['loc_city_idxs']
                query_idxs = query['query_key_idxs']
                tmp_datas.append((cuid, city_idxs[0], city_idxs, query_idxs))
        return tmp_datas