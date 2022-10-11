import numpy as np
np.random.seed(1234)
import os
from time import time
import csv
import re
import keras.backend as K
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=30, inter_op_parallelism_threads=30)))
from keras.engine.topology import Layer
from keras.layers import Dense, Input, Convolution1D, Embedding, GlobalMaxPooling1D, GRU, TimeDistributed, Dropout
from keras.layers.merge import Concatenate
from keras.models import Model, clone_model
from keras import initializers, regularizers, constraints
from keras.initializers import VarianceScaling, RandomUniform
from sklearn.metrics import f1_score, classification_report


def f1(y_true, y_pred, weighted=False):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    if weighted:
        return f1_macro, f1_micro, f1_weighted
    return f1_macro, f1_micro

def ConvolutionLayer(input_shape, n_classes, filter_sizes=[2, 3, 4, 5], num_filters=20, word_trainable=False, vocab_sz=None,
                     embedding_matrix=None, word_embedding_dim=100, hidden_dim=20, act='relu', init='ones'):
    x = Input(shape=(input_shape,), name='input')
    z = Embedding(vocab_sz, word_embedding_dim, input_length=(input_shape,), name="embedding", 
                    weights=[embedding_matrix], trainable=word_trainable)(x)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation=act,
                             strides=1,
                             kernel_initializer=init)(z)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    # z = Dropout(0.5)(z)
    z = Dense(hidden_dim, activation="relu")(z)
    y = Dense(n_classes, activation="softmax")(z)
    return Model(inputs=x, outputs=y, name='classifier')


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 init='glorot_uniform', bias=True, **kwargs):

        self.supports_masking = True
        self.init = init

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def HierAttLayer(input_shape, n_classes, word_trainable=False, vocab_sz=None,
                embedding_matrix=None, word_embedding_dim=100, gru_dim=100, fc_dim=100):
    sentence_input = Input(shape=(input_shape[2],), dtype='int32')
    embedded_sequences = Embedding(vocab_sz,
                                    word_embedding_dim,
                                    input_length=input_shape[2],
                                    weights=[embedding_matrix],
                                    trainable=word_trainable)(sentence_input)
    l_lstm = GRU(gru_dim, return_sequences=True)(embedded_sequences)
    l_dense = TimeDistributed(Dense(fc_dim))(l_lstm)
    l_att = AttentionWithContext()(l_dense)
    sentEncoder = Model(sentence_input, l_att)

    x = Input(shape=(input_shape[1], input_shape[2]), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(x)
    l_lstm_sent = GRU(gru_dim, return_sequences=True)(review_encoder)
    l_dense_sent = TimeDistributed(Dense(fc_dim))(l_lstm_sent)
    l_att_sent = AttentionWithContext()(l_dense_sent)
    y = Dense(n_classes, activation='softmax')(l_att_sent)
    
    return Model(inputs=x, outputs=y, name='classifier')


class WSTC(object):
    def __init__(self,
                 input_shape,
                 n_classes=None,
                 init=RandomUniform(minval=-0.01, maxval=0.01),
                 y=None,
                 model='cnn',
                 vocab_sz=None,
                 word_embedding_dim=100,
                 embedding_matrix=None
                 ):

        super(WSTC, self).__init__()

        self.input_shape = input_shape
        self.y = y
        self.n_classes = n_classes
        if model == 'cnn':
            self.classifier = ConvolutionLayer(self.input_shape[1], n_classes=n_classes,
                                                vocab_sz=vocab_sz, embedding_matrix=embedding_matrix, 
                                                word_embedding_dim=word_embedding_dim, init=init)
        elif model == 'rnn':
            self.classifier = HierAttLayer(self.input_shape, n_classes=n_classes,
                                             vocab_sz=vocab_sz, embedding_matrix=embedding_matrix, 
                                             word_embedding_dim=word_embedding_dim)
        
        self.model = self.classifier
        self.sup_list = {}

    def dep_pretrain(self, x, pretrain_labels, sup_idx=None, optimizer='adam',
                 loss='kld', epochs=200, batch_size=256, save_dir=None):

        self.classifier.compile(optimizer=optimizer, loss=loss)
        print("\nNeural model summary: ")
        self.model.summary()

        if sup_idx is not None:
            for i, seed_idx in enumerate(sup_idx):
                for idx in seed_idx:
                    self.sup_list[idx] = i

        # begin pretraining
        t0 = time()
        print('\nPretraining...')
        self.classifier.fit(x, pretrain_labels, batch_size=batch_size, epochs=epochs)
        print('Pretraining time: {:.2f}s'.format(time() - t0))
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.classifier.save_weights(save_dir + '/pretrained.h5')
            print('Pretrained model saved to {}/pretrained.h5'.format(save_dir))
        self.pretrained = True

    def pretrain(self, x, pretrain_labels, sup_idx=None, optimizer='adam',
                 loss='kld', epochs=200, batch_size=256, save_dir=None):

        self.classifier.compile(optimizer=optimizer, loss=loss)
        print("\nNeural model summary: ")
        self.model.summary()

        if sup_idx is not None:
            for i, seed_idx in enumerate(sup_idx):
                for idx in seed_idx:
                    self.sup_list[idx] = i

        minloss = 100
        # begin pretraining
        t0 = time()
        print('\nPretraining...')
        for e in range(epochs):
            # self.classifier.fit(x, pretrain_labels, batch_size=batch_size, epochs=1, validation_split=0.2)
            self.classifier.fit(x, pretrain_labels, batch_size=batch_size, epochs=1)
            result = self.classifier.evaluate(x, pretrain_labels)
            print(result, e)
            if result < minloss:
                result = minloss
                if save_dir is not None:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    self.classifier.save_weights(save_dir + '/pretrained.h5')
                    print('Pretrained model saved to {}/pretrained.h5'.format(save_dir))

        print("Loading model with min loss {}".format(minloss))
        self.classifier.load_weights(save_dir + '/pretrained.h5')
        print('Pretraining time: {:.2f}s'.format(time() - t0))
        
        # if save_dir is not None:
        #    if not os.path.exists(save_dir):
        ##        os.makedirs(save_dir)
        #    self.classifier.save_weights(save_dir + '/pretrained.h5')
        #    print('Pretrained model saved to {}/pretrained.h5'.format(save_dir))
        self.pretrained = True

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    def target_distribution(self, q, power=2):
        weight = q**power / q.sum(axis=0)
        p = (weight.T / weight.sum(axis=1)).T
        for i in self.sup_list:
            p[i] = 0
            p[i][self.sup_list[i]] = 1
        return p

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=10e4, batch_size=256, tol=0.1, power=2,
            update_interval=140, save_dir=None, save_suffix='', data=None):

        print("Training with samples count {}".format(len(x)))
        print('Update interval: {}'.format(update_interval))

        pred = self.classifier.predict(x)
        y_pred = np.argmax(pred, axis=1)
        y_pred_last = np.copy(y_pred)

        # logging file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/self_training_log_{}.csv'.format(save_suffix), 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'f1_macro', 'f1_micro'])
        logwriter.writeheader()

        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)

                y_pred = q.argmax(axis=1)
                p = self.target_distribution(q, power)
                print('\nIter {}: '.format(ite), end='')
                if y is not None:
                    f1_macro, f1_micro, f1_weighted = np.round(f1(y, y_pred, True), 5)
                    logdict = dict(iter=ite, f1_macro=f1_macro, f1_micro=f1_micro)
                    logwriter.writerow(logdict)
                    print('f1_macro = {}, f1_micro = {}, f1_weighted = {}'.format(f1_macro, f1_micro, f1_weighted))
                    #print(classification_report(y, y_pred))
                    
                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label*100, 3)))
                if ite > 0 and delta_label < tol/100:
                    print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label*100, 3), tol))
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            ite += 1

        logfile.close()

        if save_dir is not None:
            self.model.save_weights(save_dir + '/final.h5')
            print("Final model saved to: {}/final.h5".format(save_dir))
        return self.predict(x)

class WSTCMM(object):
    def __init__(self,
                 input_shape,
                 n_classes=None,
                 init=RandomUniform(minval=-0.01, maxval=0.01),
                 y=None,
                 model='cnn',
                 vocab_sz=None,
                 word_embedding_dim=100,
                 embedding_matrix=None
                 ):

        super(WSTCMM, self).__init__()

        self.input_shape = input_shape
        self.y = y
        self.n_classes = n_classes
        if model == 'cnn':
            self.classifier1 = ConvolutionLayer(self.input_shape[1], n_classes=n_classes,
                                                vocab_sz=vocab_sz, embedding_matrix=embedding_matrix, 
                                                word_embedding_dim=word_embedding_dim, init=init)
            self.classifier2 = ConvolutionLayer(self.input_shape[1], n_classes=n_classes,
                                                vocab_sz=vocab_sz, embedding_matrix=embedding_matrix, 
                                                word_embedding_dim=word_embedding_dim, init=init)
            # self.classifier2 = clone_model(self.classifier1)
        elif model == 'rnn':
            self.classifier1 = HierAttLayer(self.input_shape, n_classes=n_classes,
                                             vocab_sz=vocab_sz, embedding_matrix=embedding_matrix, 
                                             word_embedding_dim=word_embedding_dim)
            self.classifier2 = HierAttLayer(self.input_shape, n_classes=n_classes,
                                             vocab_sz=vocab_sz, embedding_matrix=embedding_matrix, 
                                             word_embedding_dim=word_embedding_dim)
        
        self.model1 = self.classifier1
        self.model2 = self.classifier2
        self.sup_list = {}

    def pretrain(self, x, pretrain_labels, sup_idx=None, optimizer='adam',
                 loss='kld', epochs=200, batch_size=256, save_dir=None):

        x1, x2 = x
        y1, y2 = pretrain_labels

        print(x1.shape, x2.shape)
        self.classifier1.compile(optimizer=optimizer, loss=loss)
        print("\nNeural model summary: ")
        self.model1.summary()

        self.classifier2.compile(optimizer=optimizer, loss=loss)
        print("\nNeural model summary: ")
        self.model2.summary()

        if sup_idx is not None:
            for i, seed_idx in enumerate(sup_idx):
                for idx in seed_idx:
                    self.sup_list[idx] = i

        minloss = 100
        # begin pretraining
        t0 = time()
        print('\nPretraining...')
        for e in range(epochs):
            self.classifier1.fit(x1, y1, batch_size=batch_size, epochs=1)
            result = self.classifier1.evaluate(x1, y1)
            print(result, e)
            #print(self.classifier1.predict(x1))
            if result < minloss:
                result = minloss
                if save_dir is not None:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    self.classifier1.save_weights(save_dir + '/pretrained1.h5')
                    print('Pretrained model saved to {}/pretrained1.h5'.format(save_dir))

        print("Pretrinaing for model 1 completed ....\n\n")
        for e in range(epochs):
            self.classifier2.fit(x2, y2, batch_size=batch_size, epochs=1)
            #print(self.classifier2.predict(x2))
            result = self.classifier2.evaluate(x2, y2)
            print(result, e)
            if result < minloss:
                result = minloss
                if save_dir is not None:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    self.classifier2.save_weights(save_dir + '/pretrained2.h5')
                    print('Pretrained model saved to {}/pretrained2.h5'.format(save_dir))

        print("Loading model with min loss {}".format(minloss))
        self.classifier1.load_weights(save_dir + '/pretrained1.h5')
        self.classifier2.load_weights(save_dir + '/pretrained2.h5')
        print('Pretraining time: {:.2f}s'.format(time() - t0))
        
        self.pretrained = True

    def load_weights(self, weights):
        raise NotImplementedError("aa")
        self.model1.load_weights(weights)
        self.model1.load_weights(weights)

    def predict(self, x):
        q = self.model1.predict(x, verbose=0)
        return q.argmax(1)

    def dep_target_distribution(self, q, power=2):
        wt = q**power
        p = wt / wt.sum(axis=1)[:, None]
        for i in self.sup_list:
            p[i] = 0
            p[i][self.sup_list[i]] = 1
        return p

    def target_distribution(self, q, power=2):
        weight = q**power / q.sum(axis=0)
        p = (weight.T / weight.sum(axis=1)).T
        for i in self.sup_list:
            p[i] = 0
            p[i][self.sup_list[i]] = 1
        return p

    def compile(self, optimizer='sgd', loss='kld'):
        self.model1.compile(optimizer=optimizer, loss=loss)
        self.model2.compile(optimizer=optimizer, loss=loss)

    def hin_eng_count(self, data):
        feh = [[0, 0] for _ in data]
        for i, sent in enumerate(data):
            txt = " ".join(sent)
            eng_only_cnt = len(re.sub(r'[^A-Za-z]+', ' ', txt).split())
            hin_only_cnt = len(re.sub(r'[^\u0900-\u097F]+', ' ', txt).split())
            if eng_only_cnt + hin_only_cnt == 0:
                feh[i] = [0.5, 0.5]
            else:
                fe = eng_only_cnt * 1.0 / (eng_only_cnt + hin_only_cnt)
                feh[i] = [fe, 1-fe]

        return np.array(feh)

    def fit(self, x, y=None, maxiter=10e4, batch_size=256, tol=0.1, power=2,
            update_interval=140, save_dir=None, save_suffix='', data=None):

        print('Update interval: {}'.format(update_interval))

        pred = self.classifier1.predict(x)
        y_pred_e = np.argmax(pred, axis=1)
        y_pred1_prev = np.copy(y_pred_e)
        print(classification_report(y, y_pred_e))
        pred = self.classifier2.predict(x)
        y_pred_h = np.argmax(pred, axis=1)
        print(classification_report(y, y_pred_h))
                    
        assert(data is not None)
        
        frac_eng_hin_wrds = self.hin_eng_count(data)

        # get bucketwise performance:
        for f1_type in ["micro", "macro", "weighted"]:
            bts = [([], [], [], []), ([], [], [], []), ([], [], [], []), ([], [], [], [])]
            for fe, y_pred_e_i, y_pred_h_i, y_i, sent in zip(frac_eng_hin_wrds[:, 0], 
                                                             y_pred_e, y_pred_h, y, data):

                if fe <= 0.25:
                    bts[0][0].append(y_pred_e_i); bts[0][1].append(y_pred_h_i); 
                    bts[0][2].append(y_i); bts[0][3].append(sent)
                elif fe <= 0.50:
                    bts[1][0].append(y_pred_e_i); bts[1][1].append(y_pred_h_i); 
                    bts[1][2].append(y_i); bts[1][3].append(sent)
                elif fe <= 0.75:
                    bts[2][0].append(y_pred_e_i); bts[2][1].append(y_pred_h_i); 
                    bts[2][2].append(y_i); bts[2][3].append(sent)
                else:
                    bts[3][0].append(y_pred_e_i); bts[3][1].append(y_pred_h_i); 
                    bts[3][2].append(y_i); bts[3][3].append(sent)

            for fe, bt in zip([0.25, 0.50, 0.75, 1.0], bts):
                f1_e = f1_score(bt[2], bt[0], average=f1_type)
                f1_h = f1_score(bt[2], bt[1], average=f1_type)
                # print(bt[3])
                print("{} f1 for frac_eng {} for e and h : {:.4f} {:.4f} {}".format(
                    f1_type, fe, f1_e, f1_h, len(bt[0])))
                #asdfgh = input()
                #print(asdfgh)
            print()

        asdfgh = input()
        print(asdfgh)

        # logging file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/self_training_log_{}.csv'.format(save_suffix), 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'f1_macro', 'f1_micro'])
        logwriter.writeheader()

        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q1 = self.model1.predict(x, verbose=0)
                q2 = self.model2.predict(x, verbose=0)

                y_pred1, y_pred_p1 = q1.argmax(axis=1), q1.max(axis=1)
                y_pred2, y_pred_p2 = q2.argmax(axis=1), q2.max(axis=1)
                q = q1 * frac_eng_hin_wrds[:, 0][:, None] + q2 * frac_eng_hin_wrds[:, 1][:, None]
                p = self.target_distribution(q, power)
                print('\nIter {}: '.format(ite), end='')
                if y is not None:
                    f1_macro, f1_micro, f1_weighted = np.round(f1(y, y_pred1, True), 5)
                    logdict = dict(iter=ite, f1_macro=f1_macro, f1_micro=f1_micro)
                    logwriter.writerow(logdict)
                    print('f1_macro = {}, f1_micro = {}, f1_weighted = {}'.format(f1_macro, f1_micro, f1_weighted))
                    
                # check stop criterion
                delta_label  = np.sum(y_pred1 != y_pred2).astype(np.float) / y_pred1.shape[0]
                delta_label2 = np.sum(y_pred1_prev != y_pred1).astype(np.float) / y_pred1.shape[0]
                y_pred1_prev = np.copy(y_pred1)
                
                kkkkk = 0
                for aaa, bbb, ccc, ddd, eee, fff in zip(y_pred1 != y_pred2, data, 
                                                        zip(y_pred1.tolist(), y_pred2.tolist()),
                                                        zip(y_pred_p1, y_pred_p2),
                                                        zip(q1.tolist(), q2.tolist(),
                                                            frac_eng_hin_wrds.tolist(), 
                                                            q.tolist(), 
                                                            p.tolist()), 
                                                        y):
                    if aaa:
                        print(bbb)
                        print(ccc, ddd)
                        print(eee, fff)
                        print()
                        kkkkk += 1
                        if kkkkk > 1000:
                            break
                
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label*100, 3)))
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label2*100, 3)))
                if ite > 0 and delta_label < tol/10:
                    print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label*100, 3), tol))
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            self.model1.train_on_batch(x=x[idx], y=p[idx])
            self.model2.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            ite += 1

        logfile.close()

        if save_dir is not None:
            self.model1.save_weights(save_dir + '/final1.h5')
            self.model2.save_weights(save_dir + '/final2.h5')
            print("Final model saved to: {}/final.h5".format(save_dir))
        return self.predict(x)
