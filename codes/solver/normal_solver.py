__author__ = 'SongJun-Dell'
import os, json, time, sys
import numpy as np
import cPickle as pickle
from ml_idiot.solver.BasicSolver import BasicSolver

def get_mrr(ranks):
    rs = np.array(ranks, dtype=np.float)
    mrr = np.mean( 1.0 / rs )
    mean_rank = np.mean(rs)
    return mrr, mean_rank

class NormalSolver(BasicSolver):
    def __init__(self, state):
        super(NormalSolver, self).__init__(state)
        # self.log_file = open(os.path.join(state['out_folder'], 'log.log'), 'wb')
        self.last_loss = np.inf
        self.top_valid_cost = np.inf
        self.top_valid_mrr = 0
        self.top_valid_mrank = np.inf

        self.batch_size = state['batch_size']
        self.max_epoch = state['max_epoch']
        self.valid_epoch = state['valid_epoch']
        self.valid_batch_size = state['valid_batch_size']
        self.cost_scale = state['cost_scale']

    def train(self, model, data_provider, state):
        if not os.path.exists(state['checkpoint_out_dir']):
            message = 'creating folder %s' % state['checkpoint_out_dir']
            self.log_message(message)
            os.makedirs(state['checkpoint_out_dir'])

        # log state
        self.log_message(json.dumps(state, indent=2))
        # calculate how many iteration
        train_size = data_provider.split_size('train')
        # valid_iter = max(1, int(num_iter_per_epoch * self.valid_epoch))
        valid_sample_num = max(1, int(train_size * self.valid_epoch))
        valid_sample_count = 0

        iter_count = 0
        valid_count = 0
        verbos_iter = 0
        for epoch_i in xrange(self.max_epoch):
            sample_count = 0
            for batch_data in data_provider.iter_split_running_batches(self.batch_size, split='train'):
                iter_count += 1
                t0 = time.time()
                # print 'batch size: %d' % len(batch_data)
                if iter_count == verbos_iter:
                    model.print_params()
                loss_cost = model.train(batch_data)

                if iter_count == verbos_iter:
                    print model.grad_params
                loss_cost *= self.cost_scale
                sample_count += len(batch_data)
                valid_sample_count += len(batch_data)
                # calculate smooth cost
                if iter_count == 1:
                    self.smooth_train_cost = loss_cost
                else:
                    self.smooth_train_cost = 0.99 * self.smooth_train_cost + 0.01 * loss_cost

                # print message
                time_eclipse = time.time() - t0
                # epoch_rate = 1.0 * iter_count / num_iter_per_epoch
                epoch_rate = epoch_i + 1.0 * sample_count / train_size
                message = 'samples %d/%d done in %.3fs. epoch %.3f/%d. loss_cost= %f, (smooth %f)' \
                    % (sample_count, train_size, time_eclipse, epoch_rate, self.max_epoch, loss_cost, self.smooth_train_cost)
                self.log_message(message)

                # detect loss exploding
                if not self.detect_loss_explosion(loss_cost):
                    sys.exit()

                # validation
                # if (iter_count % valid_iter) == 0 or (iter_count == max_iter):
                if (valid_sample_count >= valid_sample_num) or (epoch_i>=self.max_epoch-1 and sample_count>=train_size):
                    valid_sample_count = 0
                    valid_count += 1
                    # validate on the train valid data set
                    t0 = time.time()
                    res = self.test_on_split(model, data_provider, 'train_valid')
                    valid_num = res['sample_num']
                    valid_loss_cost = res['loss_cost']
                    # valid_loss_cost *= self.batch_size
                    valid_ranks = res['ranks']
                    mrr, mean_rank = get_mrr(valid_ranks)
                    self.get_rks(valid_ranks, [1,5,10,15,20])
                    if valid_count == 1:
                        smooth_train_valid_loss = valid_loss_cost
                    else:
                        smooth_train_valid_loss = 0.99 * smooth_train_valid_loss + 0.01 * valid_loss_cost
                    time_eclipse = time.time() - t0
                    message = 'evaluate %d train_valid samples in %.3fs and get cost %f (smooth %f) and mrr: %f, mean_rank: %f.'\
                                % (valid_num, time_eclipse, valid_loss_cost, smooth_train_valid_loss, mrr, mean_rank)
                    self.log_message(message)

                    # validate on the validate data set
                    t0 = time.time()
                    res = self.test_on_split(model, data_provider, 'valid')
                    valid_num = res['sample_num']
                    valid_loss_cost = res['loss_cost']
                    # valid_loss_cost *= self.batch_size
                    valid_ranks = res['ranks']
                    mrr, mean_rank = get_mrr(valid_ranks)
                    self.get_rks(valid_ranks, [1,5,10,15,20])
                    if valid_count == 1:
                        smooth_valid_loss = valid_loss_cost
                    else:
                        smooth_valid_loss = 0.99 * smooth_valid_loss + 0.01 * valid_loss_cost
                    time_eclipse = time.time() - t0
                    message = 'evaluate %d valid samples in %.3fs and get cost %f (smooth %f) and mrr: %f, mean_rank: %f.'\
                                % (valid_num, time_eclipse, valid_loss_cost, smooth_valid_loss, mrr, mean_rank)
                    self.log_message(message)

                    # validate on the test data set
                    t0 = time.time()
                    res = self.test_on_split(model, data_provider, 'test')
                    valid_num = res['sample_num']
                    valid_loss_cost = res['loss_cost']
                    # valid_loss_cost *= self.batch_size
                    valid_ranks = res['ranks']
                    mrr, mean_rank = get_mrr(valid_ranks)
                    self.get_rks(valid_ranks, [1,5,10,15,20])
                    if valid_count == 1:
                        smooth_test_loss = valid_loss_cost
                    else:
                        smooth_test_loss = 0.99 * smooth_test_loss + 0.01 * valid_loss_cost
                    time_eclipse = time.time() - t0
                    message = 'evaluate %d test samples in %.3fs and get cost %f (smooth %f) and mrr: %f, mean_rank: %f.'\
                                % (valid_num, time_eclipse, valid_loss_cost, smooth_test_loss, mrr, mean_rank)
                    self.log_message(message)

                    self.log_file.flush()

                    save_tag = False
                    if mrr > self.top_valid_mrr:
                        self.top_valid_mrr = mrr
                        model_file_name = 'model_checkpoint_%s_%s_mrr_%f.pkl' % (state['model_name'], state['data_set_name'], mrr)
                        state_file_name =  'state_checkpoint_%s_%s_mrr_%f.pkl' % (state['model_name'], state['data_set_name'], mrr)
                        save_tag = True
                    elif valid_loss_cost < self.top_valid_cost:
                        self.top_valid_cost = valid_loss_cost
                        model_file_name = 'model_checkpoint_%s_%s_loss_%f.pkl' % (state['model_name'], state['data_set_name'], valid_loss_cost)
                        state_file_name =  'state_checkpoint_%s_%s_loss_%f.pkl' % (state['model_name'], state['data_set_name'], valid_loss_cost)
                        save_tag = True
                    elif mean_rank < self.top_valid_mrank:
                        self.top_valid_mrank = mean_rank
                        model_file_name = 'model_checkpoint_%s_%s_mrank_%f.pkl' % (state['model_name'], state['data_set_name'], mean_rank)
                        state_file_name =  'state_checkpoint_%s_%s_mrank_%f.pkl' % (state['model_name'], state['data_set_name'], mean_rank)
                        save_tag = True
                    if save_tag:
                        model_file_path = os.path.join(state['checkpoint_out_dir'], model_file_name)
                        state_file_path = os.path.join(state['checkpoint_out_dir'], state_file_name)
                        model.save(model_file_path)
                        message = 'save checkpoint model in %s.' % model_file_path
                        self.log_message(message)
                        check_point = {
                            'iter_count': iter_count,
                            'epoch': epoch_i,
                            'state': state,
                            'valid_loss_cost': valid_loss_cost
                        }
                        pickle.dump(check_point, open(state_file_path, 'wb'))
                        message = 'save checkpoint state in %s.' % state_file_path
                        self.log_message(message)
        return model_file_path

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
            batch_loss, batch_ranks = model.test(batch_data)
            loss_cost += batch_loss * self.cost_scale
            ranks += batch_ranks
        loss_cost /= batch_num
        return {'sample_num': sample_num,
                'batch_num': batch_num,
                'ranks': ranks,
                'loss_cost': loss_cost
                }

    def log_message(self, message):
        print message
        if message[-1] != '\n': message += '\n'
        self.log_file.write(message)
        self.log_file.flush()
    def detect_loss_explosion(self, loss):
        if loss > self.smooth_train_cost * 10:
            message = 'Aborting, loss seems to exploding. try to run gradient check or lower the learning rate.'
            self.log_message(message)
            return False
        # self.smooth_train_cost = loss
        return True
    def get_rks(self, ranks, ks):
        """
        get recall @ Ks
        :param ranks: the ranks of correct index
        :param ks: values of K
        :return: print the result
        """
        rs = np.array(ranks, np.float)
        if type(ks) != list:
            ks = [ks]
        for k in ks:
            self.get_rk(rs, k)
    def get_rk(self, ranks, k):
        num = len(ranks)
        res = len(np.where(ranks<=k)[0])
        rk = res*1.0/num
        message = 'recall@ %d: %f' % (k, rk)
        self.log_message(message)
    def get_mrr(self, ranks):
        rs = np.array(ranks, dtype=np.float)
        mrr = np.mean( 1.0 / rs )
        mean_rank = np.mean(rs)
        return mrr, mean_rank
