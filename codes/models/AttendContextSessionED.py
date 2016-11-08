__author__ = 'SongJun-Dell'
from ml_idiot.nn.models.BasicModel import get_data_splits
from ContextSessionED_Model import ContextSessionED
from ml_idiot.nn.layers.Attention import Attention
import numpy as np
def get_forward_attention_caches(model, batch_data):
    res = model.get_forward_attend_cache(batch_data)
    return res

class AttendContextSessionED(ContextSessionED):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        super(AttendContextSessionED, self).__init__(state)
        self.context_attend = self.add_layer(Attention(state['context_attend_state'], rng))

    @staticmethod
    def fullfill_state(state):
        state = ContextSessionED.fullfill_state(state)
        state['context_att_len'] = state.get('context_att_len', 3)
        state['context_attend_state'] = {
            'layer_name': 'context_attend',
            'x_size': state['session_hidden_size'],
            'h_size': state['encode_hidden_size'],
            'att_size': state['session_hidden_size']
        }

        return state

    def forward_sample(self, sample, mode='train'):
        """

        :param sample: {
            'cuid_idx': cuid_idx,
            'indexed_session_queries': [ indexed_se_query, ... ]
        }
        indexed_se_query = {
            'loc_city_idxes': [city_idx, ...],
            'query_key_idxs': [query_key_idx, ...]
        }
        :param mode:
        :return:
        """
        sample_cache = dict()
        query_idxes = self.get_query_indexes(sample)
        # encoding queries
        encoding_query_vecs, encoding_query_caches = self.forward_encoding_queries(query_idxes)
        # learn context vecs
        context_vecs, context_caches = self.forward_context_sessions(encoding_query_vecs)
        # decoding queries
        sample_preds, dec_cache = self.forward_decoding_queries(query_idxes, context_vecs)

        if mode == 'train':
            sample_cache = {
                'encoding_query_caches': encoding_query_caches,
                'context_caches': context_caches,
                'deconding_query_caches': dec_cache
            }
        return sample_preds, sample_cache

    def forward_encoding_queries(self, query_idxes):
        encoding_outs = np.zeros((len(query_idxes), self.encode_lstm.state['hidden_size']))
        encoding_query_caches = list()
        for qi in xrange(len(query_idxes)-1):
            query = query_idxes[qi]
            cuid_idxs = query['cuid_idxs']
            city_idxs = query['city_idxs']
            query_idxs = query['query_idxs']

            # embedding queries
            cuid_vecs, city_vecs, query_vecs = self.embed_layer.activate(cuid_idxs, city_idxs, query_idxs)

            # forward encoding
            con_list = [query_vecs]
            if self.encode_lstm.state['use_user']: con_list.append(cuid_vecs)
            if self.encode_lstm.state['use_location']: con_list.append(city_vecs)
            lstm_current_in_vecs = np.hstack(con_list)
            lstm_outs, encode_lstm_caches = self.encode_lstm.forward_sequence(lstm_current_in_vecs)
            
            enc_out = lstm_outs[lstm_outs.shape[0]-1:lstm_outs.shape[0], :]
            enc_cache = {
                'gth_query_idxs': query_idxs,
                'gth_cuid_idxs': cuid_idxs,
                'gth_city_idxs': city_idxs,
                'enc_lstm_cache': encode_lstm_caches
            }
            encoding_query_caches.append(enc_cache)
            encoding_outs[qi] = enc_out

        encoding_query_caches.append(None)
        return encoding_outs, encoding_query_caches

    def forward_context_sessions(self, encoding_query_vecs):
        context_vecs = np.zeros((encoding_query_vecs.shape[0], self.session_lstm.state['hidden_size']))
        context_caches = list()
        attend_cache = None
        for qi in xrange(encoding_query_vecs.shape[0]-1):
            if qi == 0:
                context_lstm_input = self.session_lstm.default_out
                prev_context_lstm_cell = self.session_lstm.default_cell
            elif qi == 1:
                context_lstm_input = prev_context_lstm_out
            elif qi > 1:
                context_lstm_input, attend_cache = self.context_attend.activate(context_vecs[max(0,qi-self.state['context_att_len']):qi,:], encoding_query_vecs[qi:qi+1,:])
            prev_context_lstm_out, prev_context_lstm_cell, context_lstm_cache = self.session_lstm.activate(encoding_query_vecs[qi:qi+1,:], context_lstm_input, prev_context_lstm_cell)
            context_vecs[qi] = prev_context_lstm_out
            context_caches.append({'context_attend_cache':attend_cache,'context_lstm_cache':context_lstm_cache})
        context_caches.append(None)

        return context_vecs, context_caches

    def forward_decoding_queries(self, query_idxes, context_vecs):
        sample_preds = list()
        decoding_query_caches = list()
        for qi,query in enumerate(query_idxes):
            cuid_idxs = query['cuid_idxs']
            city_idxs = query['city_idxs']
            query_idxs = query['query_idxs']
            in_cuid_idxs = [cuid_idxs[0]] + cuid_idxs
            in_city_idxs = [city_idxs[0]] + city_idxs
            in_query_idxs = [1] + query_idxs

            cuid_vecs, city_vecs, query_vecs = self.embed_layer.activate(in_cuid_idxs, in_city_idxs, in_query_idxs)
            if qi == 0:
                prev_context_out = self.session_lstm.default_out
            else:
                prev_context_out = context_vecs[qi-1:qi,:]

            pred_probs, dec_cache = self.forward_dec(cuid_vecs[:-1,:], city_vecs[:-1,:], query_vecs[:-1,:], prev_context_out)
            sample_preds.append({'gth_idx': query['query_idxs'], 'pred_probs': pred_probs})

            temp_cache = {
                'gth_query_idxs': query_idxs,
                'gth_cuid_idxs': cuid_idxs,
                'gth_city_idxs': city_idxs,
                'pred_probs': pred_probs
            }
            dec_cache.update(temp_cache)

            decoding_query_caches.append(dec_cache)

        return sample_preds, decoding_query_caches

    def backward_sample(self, grad_params, sample_cache_data):
        self.temp_values['session_size'] = len(sample_cache_data['deconding_query_caches'])
        # backward decoding layer
        grad_context_lstm_outs = self.backward_deconding_queries(grad_params, sample_cache_data['deconding_query_caches'])
        # backward context layer
        grad_encoding_lstm_outs = self.backward_context_sessions(grad_params, grad_context_lstm_outs, sample_cache_data['context_caches'])
        # backward encoding layer
        self.backward_encoding_queries(grad_params, grad_encoding_lstm_outs, sample_cache_data['encoding_query_caches'])

    def backward_deconding_queries(self, grad_params, dec_query_caches):
        grad_context_lstm_outs = np.zeros((len(dec_query_caches), self.session_lstm.state['hidden_size']))
        for qi,dec_query_cache in enumerate(dec_query_caches):
            pred_probs = dec_query_cache['pred_probs']
            gth_query_idxs = dec_query_cache['gth_query_idxs']
            gth_cuid_idxs = dec_query_cache['gth_cuid_idxs']
            gth_city_idxs = dec_query_cache['gth_city_idxs']
            qes = self.embed_layer.state['query_emb_size']
            cues = self.embed_layer.state['cuid_emb_size']

            # gradient of out mapping output y
            grad_y = pred_probs
            grad_y[range(len(gth_query_idxs)), gth_query_idxs] -= 1
            grad_y /= (len(gth_query_idxs)*self.temp_values['session_size']*self.temp_values['batch_size'])

            # back ward out_mapping_layer
            out_mapping_in_vecs = dec_query_cache['out_mapping_in_vecs']
            grad_dec_lstm_current_out_vecs = self.out_mapping_layer.backward(grad_params, out_mapping_in_vecs, grad_y)

            # back ward decode_lstm layer
            grad_dec_lstm_current_in_vecs = self.decode_lstm.backward_whole_sequence(grad_params, grad_dec_lstm_current_out_vecs, dec_query_cache['decode_lstm_caches'])
            grad_prev_session_lstm_out = np.sum(grad_dec_lstm_current_in_vecs[:, grad_dec_lstm_current_in_vecs.shape[1]-self.session_lstm.state['hidden_size']:],keepdims=True)
            if qi == 0:
                grad_params[self.session_lstm.default_out_name] += grad_prev_session_lstm_out
            else:
                grad_context_lstm_outs[qi-1:qi,:] = grad_prev_session_lstm_out

            # backward embed_layer
            grad_current_emb_outs = grad_dec_lstm_current_in_vecs[:, :grad_dec_lstm_current_in_vecs.shape[1]-self.session_lstm.state['hidden_size']]
            grad_query_embs = grad_current_emb_outs[:, :qes]
            grad_cuid_embs = grad_current_emb_outs[:, qes:qes+cues] if self.decode_lstm.state['use_user'] else None
            grad_city_embs = grad_current_emb_outs[:, qes+cues:] if self.decode_lstm.state['use_location'] else None
            self.embed_layer.backward(grad_params, [1]+gth_query_idxs[0:-1], [gth_cuid_idxs[0]]+gth_cuid_idxs[0:-1], [gth_city_idxs[0]]+gth_city_idxs[0:-1],
                                      grad_query_embs, grad_cuid_embs, grad_city_embs)

        return grad_context_lstm_outs

    def backward_context_sessions(self, grad_params, grad_context_lstm_outs, context_caches):
        grad_encoding_outs = np.zeros((grad_context_lstm_outs.shape[0], self.encode_lstm.state['hidden_size']))
        grad_prev_session_lstm_cell = np.zeros(self.session_lstm.default_cell.shape)
        # skip the last query
        for qi in reversed(xrange(grad_context_lstm_outs.shape[0]-1)):
            query_cache = context_caches[qi]
            grad_lstm_in, grad_prev_session_lstm_cell = self.session_lstm.backward_step(grad_params,
                                                    grad_context_lstm_outs[qi:qi+1,:], query_cache['context_lstm_cache'], grad_prev_session_lstm_cell)
            grad_encoding_outs[qi:qi+1,:] =  grad_lstm_in[:, :self.session_lstm.state['hidden_size']]
            grad_context_in = grad_lstm_in[:, :self.session_lstm.state['hidden_size']]

            if qi == 0:
                grad_params[self.session_lstm.default_out_name] += grad_context_in
                grad_params[self.session_lstm.default_cell_name] += grad_prev_session_lstm_cell
            elif qi == 1:
                grad_context_lstm_outs[qi-1:qi,:] += grad_context_in
            elif qi > 1:
                grad_att_in_x, grad_att_in_h = self.context_attend.backward_step(grad_params, grad_context_in, query_cache['context_attend_cache'])
                grad_encoding_outs[qi:qi+1,:] += grad_att_in_h
                grad_context_lstm_outs[max(0,qi-self.state['context_att_len']):qi,:] += grad_att_in_x

        return grad_encoding_outs

    def backward_encoding_queries(self, grad_params, grad_encoding_lstm_outs, enc_query_caches):
        for qi in xrange(grad_encoding_lstm_outs.shape[0]-1):
            enc_query_cache = enc_query_caches[qi]
            enc_lstm_cache = enc_query_cache['enc_lstm_cache']
            grad_encode_out = grad_encoding_lstm_outs[qi:qi+1,:]

            grad_enc_current_in_vecs = self.backward_encode_lstm_layer(grad_params, grad_encode_out, enc_lstm_cache)

            self.backward_embed_layer(grad_params, grad_enc_current_in_vecs, enc_query_cache)

    def get_forward_attend_cache(self, batch_data):
        batch_query_preds, batch_cache = self.forward_batch(batch_data, mode='train')
        batch_cost = self.get_batch_cost(batch_query_preds)
        ranks = self.get_batch_ranks(batch_query_preds)
        batch_attend_caches = list()
        for sample_cache in batch_cache:
            attend_sample_cache = list()
            context_caches = sample_cache['context_caches']

            for context_cache in context_caches:
                if context_cache is None:
                    attend_sample_cache.append({'attend_weights': 0})
                else:
                    attend_cache = context_cache['context_attend_cache']
                    if attend_cache is None:
                        attend_sample_cache.append({'attend_weights': 0})
                    else:
                        attend_sample_cache.append({'attend_weights': attend_cache['att_weight']})
            batch_attend_caches.append(attend_sample_cache)

        return {'loss': batch_cost, 'ranks': ranks, 'batch_attend_caches':batch_attend_caches}

    def get_cache(self, batch_data, pool=None, num_processes=0):
        self.set_unknow_embedding()
        if pool is None:
            res = get_forward_attention_caches(self, batch_data)
        else:
            loss = list()
            ranks = list()
            attend_caches = list()
            mini_batch_size = self.state['batch_size']
            sample_num = len(batch_data)
            print 'num of samples: %d' % sample_num
            start_pos = 0
            while start_pos < sample_num:
                print 'start position %d' % start_pos
                end_pos = start_pos + mini_batch_size
                mini_batch_data = batch_data[start_pos:end_pos]
                mini_res = self.get_mini_batch_cache(mini_batch_data, pool, num_processes)
                loss.append(mini_res['loss'])
                ranks += mini_res['ranks']
                attend_caches += mini_res['attend_caches']

                start_pos = end_pos
            loss = sum(loss)/len(loss) + self.state['regularize_rate'] * self.get_regularization()
            res = {'loss': loss, 'ranks':ranks, 'attend_caches': attend_caches}
        return res

    def get_mini_batch_cache(self, mini_batch_data, pool, num_processes):
        batch_splits = get_data_splits(num_processes, mini_batch_data)
        loss = 0
        ranks = list()
        attend_caches = list()
        pool_results = list()
        for pro in xrange(num_processes):
            data_samples = batch_splits[pro]
            tmp_result = pool.apply_async(get_forward_attention_caches, (self, data_samples))
            pool_results.append(tmp_result)
        for p in xrange(num_processes):
            split_res = pool_results[p].get()
            loss += split_res['loss']
            ranks += split_res['ranks']
            attend_caches += split_res['batch_attend_caches']
        loss /= num_processes
        res = {'loss': loss, 'ranks':ranks, 'attend_caches': attend_caches}
        return res