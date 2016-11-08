__author__ = 'SongJun-Dell'
from QueryPredModel import QueryPredModel
from ml_idiot.nn.layers.FullConnect import FullConnect
from ml_idiot.nn.layers.QEmbedding import QEmbedding
from ml_idiot.nn.layers.LSTM import LSTM
import numpy as np

class SessED(QueryPredModel):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        super(SessED, self).__init__(state)
        self.rng = rng
        self.embed_layer = self.add_layer(QEmbedding(state['qembedding_state'], rng))
        self.encode_lstm = self.add_layer(LSTM(state['encode_lstm_state'], rng))
        self.session_lstm = self.add_layer(LSTM(state['session_lstm_state'], rng))
        self.decode_lstm = self.add_layer(LSTM(state['decode_lstm_state'], rng))
        self.out_mapping_layer = self.add_layer(FullConnect(state['out_mapping_state'], rng))

        self.temp_values = dict()

        self.check()

    @staticmethod
    def fullfill_state(state):
        state['qembedding_state'] = QEmbedding.fullfill_state(state)

        state['encode_lstm_state'] = {
            'layer_name': 'encode_lstm',
            'hidden_size': state['encode_hidden_size'],
            'lstm_input_size': state['query_emb_size'] + state['encode_hidden_size'],
            'use_user': state['use_user'],
            'use_location': state['use_location'],
            'gate_act': state.get('encode_lstm_gate_act', state['lstm_gate_act']),
            'hidden_act': state.get('encode_lstm_hidden_act', state['lstm_hidden_act'])
        }

        assert(state['session_hidden_size']==state['query_emb_size'], 'session hidden size not equal to query emb size')
        state['session_lstm_state'] = {
            'layer_name': 'session_lstm',
            'hidden_size': state['session_hidden_size'],
            'lstm_input_size': state['encode_hidden_size'] + state['session_hidden_size'],
            'gate_act': state.get('session_lstm_gate_act', state['lstm_gate_act']),
            'hidden_act': state.get('session_lstm_hidden_act', state['lstm_hidden_act'])
        }

        state['decode_lstm_state'] = {
            'layer_name': 'decode_lstm',
            'hidden_size': state['decode_hidden_size'],
            'lstm_input_size': state['session_hidden_size'] + state['decode_hidden_size'],
            'use_user': state['use_user'],
            'use_location': state['use_location'],
            'gate_act': state.get('decode_lstm_gate_act', state['lstm_gate_act']),
            'hidden_act': state.get('decode_lstm_hidden_act', state['lstm_hidden_act'])
        }

        state['out_mapping_state'] = {
            'layer_name': 'out_mapping',
            'input_size': state['decode_hidden_size'],
            'output_size': state['query_num'],
            'activation_func':'softmax'
        }

        if state['use_user']:
            state['encode_lstm_state']['lstm_input_size'] += state['cuid_emb_size']
            state['decode_lstm_state']['lstm_input_size'] += state['cuid_emb_size']
        if state['use_location']:
            state['encode_lstm_state']['lstm_input_size'] += state['city_emb_size']
            state['decode_lstm_state']['lstm_input_size'] += state['city_emb_size']

        return state

    def get_cost(self, batch_data, mode='test'):
        cost, train_cache = self.cost_func(batch_data)
        if mode == 'train':
            grad_params = self.calculate_grads(train_cache)
            return cost, grad_params
        elif mode == 'test':
            return cost
        else:
            raise StandardError('mode error')

    def cost_func(self, batch_data):
        batch_query_preds, train_batch_cache = self.forward_batch(batch_data, mode='train')
        batch_cost = self.get_batch_cost(batch_query_preds)
        cost = batch_cost + self.state['regularize_rate'] * self.get_regularization()
        return cost, train_batch_cache

    def backward_sample(self, grad_params, sample_cache_data):
        grad_cur_session_lstm_out = None
        grad_cur_session_cell = None
        self.temp_values['session_size'] = len(sample_cache_data)
        for qi in reversed(xrange(len(sample_cache_data))):
            query_cache_data = sample_cache_data[qi]
            grad_cur_session_lstm_out, grad_cur_session_cell = self.backward_query(grad_params, query_cache_data, grad_cur_session_lstm_out, grad_cur_session_cell)
        # print grad_cur_session_cell.shape
        # print grad_cur_session_lstm_out.shape
        grad_params[self.session_lstm.default_out_name] += grad_cur_session_lstm_out
        grad_params[self.session_lstm.default_cell_name] += grad_cur_session_cell

    def backward_query(self, grad_params, query_cache_data, grad_cur_session_lstm_out, grad_session_cell):
        pred_probs = query_cache_data['pred_probs']
        gth_query_idxs = query_cache_data['gth_query_idxs']
        gth_cuid_idxs = query_cache_data['gth_cuid_idxs']
        gth_city_idxs = query_cache_data['gth_city_idxs']
        enc_lstm_caches = query_cache_data['encode_lstm_caches']
        qes = self.embed_layer.state['query_emb_size']
        cues = self.embed_layer.state['cuid_emb_size']
        if grad_cur_session_lstm_out is not None:
            # backward session_lstm layer
            grad_encode_out, grad_prev_session_lstm_out, grad_prev_session_lstm_cell = \
                self.backward_session_lstm_layer(grad_params, grad_cur_session_lstm_out, query_cache_data['session_lstm_cache'], grad_session_cell)

            # backward encode_lstm layer
            grad_recurrent_in_vecs = self.backward_encode_lstm_layer(grad_params, grad_encode_out, enc_lstm_caches)

            # backward embed_layer
            self.backward_embed_layer(grad_params, grad_recurrent_in_vecs, query_cache_data)

        else:
            grad_prev_session_lstm_out = np.zeros(query_cache_data['session_lstm_cache']['cell'].shape)
            grad_prev_session_lstm_cell = np.zeros(query_cache_data['session_lstm_cache']['cell'].shape)

        # gradient of out mapping output y
        grad_y = pred_probs
        grad_y[range(len(gth_query_idxs)), gth_query_idxs] -= 1
        grad_y /= (len(gth_query_idxs)*self.temp_values['session_size']*self.temp_values['batch_size'])

        # back ward out_mapping_layer
        out_mapping_in_vecs = query_cache_data['out_mapping_in_vecs']
        grad_dec_lstm_current_out_vecs = self.out_mapping_layer.backward(grad_params, out_mapping_in_vecs, grad_y)

        # back ward decode_lstm layer
        grad_dec_lstm_current_in_vecs = self.decode_lstm.backward_whole_sequence(grad_params, grad_dec_lstm_current_out_vecs, query_cache_data['decode_lstm_caches'])
        grad_prev_session_lstm_out += grad_dec_lstm_current_in_vecs[0:1,:self.session_lstm.state['hidden_size']]

        # backward embed_layer
        grad_query_embs = grad_dec_lstm_current_in_vecs[1:, :qes]
        grad_cuid_embs = grad_dec_lstm_current_in_vecs[:, qes:qes+cues] if self.decode_lstm.state['use_user'] else None
        grad_city_embs = grad_dec_lstm_current_in_vecs[:, qes+cues:] if self.decode_lstm.state['use_location'] else None
        self.embed_layer.backward(grad_params, gth_query_idxs[0:-1], [gth_cuid_idxs[0]]+gth_cuid_idxs[0:-1], [gth_city_idxs[0]]+gth_city_idxs[0:-1], grad_query_embs, grad_cuid_embs, grad_city_embs)

        return grad_prev_session_lstm_out, grad_prev_session_lstm_cell

    def backward_session_lstm_layer(self, grad_params, grad_cur_session_lstm_out, query_session_lstm_cache, grad_session_cell):
        grad_lstm_in, grad_prev_session_lstm_cell = self.session_lstm.backward_step(grad_params, grad_cur_session_lstm_out, query_session_lstm_cache, grad_session_cell)
        grad_encode_out = grad_lstm_in[:, self.session_lstm.state['hidden_size']:]
        grad_prev_session_lstm_out = grad_lstm_in[:, :self.session_lstm.state['hidden_size']]
        return grad_encode_out, grad_prev_session_lstm_out, grad_prev_session_lstm_cell

    def backward_encode_lstm_layer(self, grad_params, grad_encode_out, enc_lstm_caches):
        grad_encode_hidden_out_vecs = np.zeros((len(enc_lstm_caches), self.encode_lstm.state['hidden_size']))
        grad_encode_hidden_out_vecs[grad_encode_hidden_out_vecs.shape[0]-1:grad_encode_hidden_out_vecs.shape[0],:] += grad_encode_out
        grad_recurrent_in_vecs = self.encode_lstm.backward_whole_sequence(grad_params, grad_encode_hidden_out_vecs, enc_lstm_caches)
        return grad_recurrent_in_vecs

    def backward_embed_layer(self, grad_params, grad_recurrent_in_vecs, query_cache_data):
        gth_query_idxs = query_cache_data['gth_query_idxs']
        gth_cuid_idxs = query_cache_data['gth_cuid_idxs']
        gth_city_idxs = query_cache_data['gth_city_idxs']
        qes = self.embed_layer.state['query_emb_size']
        cues = self.embed_layer.state['cuid_emb_size']

        grad_query_embs = grad_recurrent_in_vecs[:, :qes]

        grad_cuid_embs = grad_recurrent_in_vecs[:, qes:qes+cues] if self.encode_lstm.state['use_user'] else None

        grad_city_embs = grad_recurrent_in_vecs[:, qes+cues:] if self.encode_lstm.state['use_location'] else None

        self.embed_layer.backward(grad_params, gth_query_idxs, gth_cuid_idxs, gth_city_idxs, grad_query_embs, grad_cuid_embs, grad_city_embs)

    def forward_sample(self, sample, mode='train'):
        sample_cache = list()
        query_indexes = self.get_query_indexes(sample)
        prev_session_lstm_out = self.session_lstm.default_out
        prev_session_lstm_cell = self.session_lstm.default_cell
        sample_preds = list()
        for query in query_indexes:
            pred_probs, prev_session_lstm_out, prev_session_lstm_cell, query_cahce = self.forward_query(query, prev_session_lstm_out, prev_session_lstm_cell, mode)
            if mode=='train':
                sample_cache.append(query_cahce)
            sample_preds.append({'gth_idx': query['query_idxs'], 'pred_probs': pred_probs})
        return sample_preds, sample_cache

    def forward_query(self, query, prev_session_out, prev_session_cell, mode='train'):
        # decode query
        cuid_idxs = query['cuid_idxs']
        city_idxs = query['city_idxs']
        query_idxs = query['query_idxs']
        in_cuid_idxs = [cuid_idxs[0]] + cuid_idxs
        in_city_idxs = [city_idxs[0]] + city_idxs
        query_cache = dict()

        cuid_vecs, city_vecs, query_vecs = self.embed_layer.activate(in_cuid_idxs, in_city_idxs, query_idxs)

        query_vecs = np.vstack([prev_session_out, query_vecs])

        pred_probs, dec_cache = self.forward_dec(cuid_vecs[:-1, :], city_vecs[:-1, :], query_vecs[:-1, :]) # (query_size, vocab_size)

        # encode query
        new_session_out, new_session_cell, enc_cache = self.forward_enc(cuid_vecs[1:, :], city_vecs[1:, :], query_vecs[1:, :], prev_session_out, prev_session_cell)
        if mode=='train':
            query_cache = {
                'gth_query_idxs': query_idxs,
                'gth_cuid_idxs': cuid_idxs,
                'gth_city_idxs': city_idxs,
                'pred_probs': pred_probs
            }
            query_cache.update(dec_cache)
            query_cache.update(enc_cache)

        return pred_probs, new_session_out, new_session_cell, query_cache

    def forward_dec(self, cuid_vecs, city_vecs, query_vecs):
        con_list = [query_vecs]
        if self.decode_lstm.state['use_user']: con_list.append(cuid_vecs)
        if self.decode_lstm.state['use_location']: con_list.append(city_vecs)
        lstm_current_in_vecs = np.hstack(con_list)

        lstm_outs, decode_lstm_caches = self.decode_lstm.forward_sequence(lstm_current_in_vecs)
        pred_probs = self.out_mapping_layer.activate(lstm_outs)
        dec_cache = {
            'decode_lstm_caches':decode_lstm_caches,
            'out_mapping_in_vecs': lstm_outs
        }
        return pred_probs, dec_cache

    def forward_enc(self, cuid_vecs, city_vecs, query_vecs, prev_session_out, prev_session_cell):
        # forward encode layer
        con_list = [query_vecs]
        if self.encode_lstm.state['use_user']: con_list.append(cuid_vecs)
        if self.encode_lstm.state['use_location']: con_list.append(city_vecs)
        lstm_current_in_vecs = np.hstack(con_list)
        lstm_outs, encode_lstm_caches = self.encode_lstm.forward_sequence(lstm_current_in_vecs)

        # forward session_lstm layer
        enc_out = lstm_outs[lstm_outs.shape[0]-1:lstm_outs.shape[0], :]
        new_session_out, new_session_cell, session_lstm_cache = self.session_lstm.activate(enc_out, prev_session_out, prev_session_cell)
        enc_cache = {
            'encode_lstm_caches': encode_lstm_caches,
            'session_lstm_cache': session_lstm_cache
        }
        return new_session_out, new_session_cell, enc_cache

    def get_sample_cost(self, sample_preds):
        sample_cost = list()
        for query_preds in sample_preds:
            query_cost = self.get_query_cost(query_preds)
            sample_cost.append(query_cost)
        return np.mean(sample_cost)

    def get_query_cost(self, query_preds):
        gth_idx = query_preds['gth_idx']
        pred_probs = query_preds['pred_probs']
        probs = list()
        for ki, kidx in enumerate(gth_idx):
            probs.append(pred_probs[ki, kidx])
        query_cost = -np.mean(np.log(np.array(probs) + 1e-20))
        return query_cost

    def get_query_indexes(self, sample):
        query_indexes = list()
        cuid_idx = sample['cuid_idx']
        session_queries = sample['indexed_session_queries']
        for query in session_queries:
            city_idxs = query['loc_city_idxs']
            query_idxs = query['query_key_idxs']
            cuid_idxs = list(np.ones(len(query_idxs)).astype(int) * cuid_idx)
            query_indexes.append({'cuid_idxs':cuid_idxs, 'city_idxs':city_idxs, 'query_idxs':query_idxs})
        return query_indexes

    def set_unknow_embedding(self):
        if self.state['use_user']:
            self.embed_layer.cuid_emb_matrix[self.state['cuid_unknow_idx']].setfield(np.mean(self.embed_layer.cuid_emb_matrix, axis=0),
                                                                                     self.embed_layer.cuid_emb_matrix.dtype)
        if self.state['use_location']:
            self.embed_layer.city_emb_matrix[self.state['city_unknow_idx']].setfield(np.mean(self.embed_layer.city_emb_matrix, axis=0),
                                                                                     self.embed_layer.city_emb_matrix.dtype)

        self.embed_layer.query_emb_matrix[self.state['query_unknow_idx']].setfield(np.mean(self.embed_layer.query_emb_matrix, axis=0),
                                                                                   self.embed_layer.query_emb_matrix.dtype)