__author__ = 'SongJun-Dell'

from SessionED_Model import SessED
from ml_idiot.nn.layers.QEmbedding import QEmbedding
import numpy as np

class ContextSessionED(SessED):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        super(ContextSessionED, self).__init__(state)

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
            'lstm_input_size': state['query_emb_size'] + state['session_hidden_size'] + state['decode_hidden_size'],
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

    def forward_query(self, query, prev_session_out, prev_session_cell, mode='train'):
        # decode query
        cuid_idxs = query['cuid_idxs']
        city_idxs = query['city_idxs']
        query_idxs = query['query_idxs']
        in_cuid_idxs = [cuid_idxs[0]] + cuid_idxs
        in_city_idxs = [city_idxs[0]] + city_idxs
        in_query_idxs = [1] + query_idxs

        cuid_vecs, city_vecs, query_vecs = self.embed_layer.activate(in_cuid_idxs, in_city_idxs, in_query_idxs)

        # forward decode
        pred_probs, dec_cache = self.forward_dec(cuid_vecs[:-1,:], city_vecs[:-1,:], query_vecs[:-1,:], prev_session_out)

        # encode query
        new_session_out, new_session_cell, enc_cache = self.forward_enc(cuid_vecs[1:, :], city_vecs[1:, :], query_vecs[1:, :], prev_session_out, prev_session_cell)

        query_cache = dict()
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

    def forward_dec(self, cuid_vecs, city_vecs, query_vecs, prev_session_out):
        con_list = [query_vecs]
        if self.decode_lstm.state['use_user']: con_list.append(cuid_vecs)
        if self.decode_lstm.state['use_location']: con_list.append(city_vecs)
        con_list.append(np.tile(prev_session_out, (query_vecs.shape[0], 1)))
        lstm_current_in_vecs = np.hstack(con_list)

        lstm_outs, decode_lstm_caches = self.decode_lstm.forward_sequence(lstm_current_in_vecs)
        pred_probs = self.out_mapping_layer.activate(lstm_outs)
        dec_cache = {
            'decode_lstm_caches': decode_lstm_caches,
            'out_mapping_in_vecs': lstm_outs
        }
        return pred_probs, dec_cache

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
        grad_prev_session_lstm_out += np.sum(grad_dec_lstm_current_in_vecs[:, grad_dec_lstm_current_in_vecs.shape[1]-self.session_lstm.state['hidden_size']:],keepdims=True)

        # backward embed_layer
        grad_current_emb_outs = grad_dec_lstm_current_in_vecs[:, :grad_dec_lstm_current_in_vecs.shape[1]-self.session_lstm.state['hidden_size']]
        grad_query_embs = grad_current_emb_outs[:, :qes]
        grad_cuid_embs = grad_current_emb_outs[:, qes:qes+cues] if self.decode_lstm.state['use_user'] else None
        grad_city_embs = grad_current_emb_outs[:, qes+cues:] if self.decode_lstm.state['use_location'] else None
        self.embed_layer.backward(grad_params, [1]+gth_query_idxs[0:-1], [gth_cuid_idxs[0]]+gth_cuid_idxs[0:-1], [gth_city_idxs[0]]+gth_city_idxs[0:-1],
                                  grad_query_embs, grad_cuid_embs, grad_city_embs)

        return grad_prev_session_lstm_out, grad_prev_session_lstm_cell

    def load_from_dump(self, model_dump):
        model_params = model_dump['params']
        if 'encode_lstm_default_cell' not in model_params:
            hidden_size = model_dump['state']['hidden_size']
            model_params['encode_lstm_default_cell'] = model_params['encode_lstm_init_hidden_state'][:,:hidden_size]
            model_params['encode_lstm_default_out'] = model_params['encode_lstm_init_hidden_state'][:,hidden_size:]

            model_params['decode_lstm_default_cell'] = model_params['decode_lstm_init_hidden_state'][:,:hidden_size]
            model_params['decode_lstm_default_out'] = model_params['decode_lstm_init_hidden_state'][:,hidden_size:]

            model_params['session_lstm_default_cell'] = model_params['session_lstm_init_hidden_state'][:,:hidden_size]
            model_params['session_lstm_default_out'] = model_params['session_lstm_init_hidden_state'][:,hidden_size:]

        for p in self.params.keys():
            self.params[p].setfield(model_params[p], dtype=self.params[p].dtype)
        self.state = model_dump['state']