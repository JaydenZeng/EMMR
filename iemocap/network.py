import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
from Settings import Config
from module import ff, multihead_attention, ln, mask, SigmoidAtt
import sys
from tensorflow.python.keras.utils import losses_utils



class MM:
    def __init__(self, is_training):
        self.config = Config()
        self.att_dim = self.config.att_dim
        self.visual = tf.placeholder(dtype = tf.float32, shape=[self.config.batch_size, self.config.max_visual_len, 709], name='visual')
        self.audio = tf.placeholder(dtype = tf.float32, shape = [self.config.batch_size, self.config.max_audio_len, 33], name='audio')
        self.text = tf.placeholder(dtype = tf.float32, shape = [self.config.batch_size, self.config.max_text_len, 768], name='text')
        self.label = tf.placeholder(dtype = tf.int32, shape = [self.config.batch_size], name = 'label')
        self.flag = tf.placeholder(dtype = tf.int32, shape = [self.config.batch_size], name = 'flag')
        self.pretrained_output = tf.placeholder(dtype = tf.float32, shape = [self.config.batch_size, 300, 300], name = 'pre')

        with tf.variable_scope('KMM'):
            self.preprocess()
            #trans
            trans_en, trans_de = self.trans()
            trans_v, trans_a, trans_t = self.recovery(trans_de)
            trans_multi = self.multi_res(trans_en, trans_v, trans_a, trans_t, 'trans_wei')

            #ae
            ae_en, ae_de = self.AE()
            ae_v, ae_a, ae_t = self.recovery(ae_de)
            ae_multi = self.multi_res(ae_en, ae_v, ae_a, ae_t, 'ae_wei')

            #mmin
            mmin_en, mmin_de = self.MMIN()
            mmin_v, mmin_a, mmin_t = self.recovery(mmin_de)
            mmin_multi = self.multi_res(mmin_en, mmin_v, mmin_a, mmin_t, 'mmin_wei')

            #ensemble
            final_res = self.ensemble(trans_multi, ae_multi, mmin_multi)
            self.cal_loss(final_res, trans_en, trans_de)
            #self.cal_loss(final_res, ae_en, ae_de)
            #self.cal_loss(final_res, mmin_en, mmin_de)

        trans_en = tf.multiply(trans_en, 1, name='encode_outputs')
        ae_en = tf.multiply(ae_en, 1, name='ae_en')
        ae_de = tf.multiply(ae_de, 1, name='ae_de')
        mmin_en = tf.multiply(mmin_en, 1, name='mmin_en')
        mmin_de = tf.multiply(mmin_de, 1, name='mmin_de')

    def preprocess(self):
        visual = tf.layers.dense(self.visual, self.config.att_dim, use_bias=False)
        audio = tf.layers.dense(self.audio, self.config.att_dim, use_bias=False)
        text = tf.layers.dense(self.text, self.config.att_dim, use_bias =False)
        
        with tf.variable_scope('vv', reuse=tf.AUTO_REUSE):
          enc_vv = multihead_attention(queries=visual,
                                   keys=visual,
                                   values=visual,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          self.enc_vv = ff(enc_vv, num_units=[4*self.config.att_dim, self.config.att_dim])


        with tf.variable_scope('aa', reuse=tf.AUTO_REUSE):
          enc_aa = multihead_attention(queries=audio,
                                   keys=audio,
                                   values=audio,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          self.enc_aa = ff(enc_aa, num_units=[4*self.config.att_dim, self.config.att_dim])

        with tf.variable_scope('tt', reuse=tf.AUTO_REUSE):
          enc_tt = multihead_attention(queries=text,
                                   keys=text,
                                   values=text,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          self.enc_tt = ff(enc_tt, num_units=[4*self.config.att_dim, self.config.att_dim])


        with tf.variable_scope('common_weights', reuse = tf.AUTO_REUSE):
          wei_va = tf.get_variable('wei_va', [self.att_dim, 150])
          wei_vt = tf.get_variable('wei_vt', [self.att_dim, 150])
          wei_ta = tf.get_variable('wei_ta', [self.att_dim, 150])


        self.common_v = tf.reshape(tf.concat([tf.matmul(tf.reshape(enc_vv, [-1, self.att_dim]), wei_va), tf.matmul(tf.reshape(enc_vv, [-1, self.att_dim]), wei_vt)], -1), [self.config.batch_size, -1, self.att_dim])
        self.common_a = tf.reshape(tf.concat([tf.matmul(tf.reshape(enc_aa, [-1, self.att_dim]), wei_va), tf.matmul(tf.reshape(enc_aa, [-1, self.att_dim]), wei_ta)], -1), [self.config.batch_size, -1, self.att_dim])
        self.common_t = tf.reshape(tf.concat([tf.matmul(tf.reshape(enc_tt, [-1, self.att_dim]), wei_vt), tf.matmul(tf.reshape(enc_tt, [-1, self.att_dim]), wei_ta)], -1), [self.config.batch_size, -1, self.att_dim])
        self.enc_all = tf.concat([self.common_v, self.common_a, self.common_t], 1)


        self.enc_new = tf.convert_to_tensor(self.enc_all)



    def trans(self):
        with tf.variable_scope('nn', reuse=tf.AUTO_REUSE):
          enc_en = multihead_attention(queries=self.enc_new,
                                   keys=self.enc_new,
                                   values=self.enc_new,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          enc_en = ff(enc_en, num_units=[4*300, 300])


        with tf.variable_scope('de', reuse=tf.AUTO_REUSE):
          enc_de = multihead_attention(queries=enc_en,
                                   keys=enc_en,
                                   values=enc_en,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          enc_de = ff(enc_de, num_units=[4*300, 300])
        return enc_en, enc_de


    def AE(self):
        ae_en = []
        ae_de = []
        tmp_en = []
        tmp_de = []
        #encoder
        layer1 = tf.nn.leaky_relu(tf.layers.dense(self.enc_new, 300, use_bias=False))
        layer2 = tf.nn.leaky_relu(tf.layers.dense(layer1, 256, use_bias=False))
        layer3 = tf.nn.leaky_relu(tf.layers.dense(layer2, 128, use_bias=False))
        layer4 = tf.nn.leaky_relu(tf.layers.dense(layer3, 64, use_bias=False))
        #decoder
        layer5 = tf.nn.leaky_relu(tf.layers.dense(layer4, 128, use_bias=False))
        layer6 = tf.nn.leaky_relu(tf.layers.dense(layer5, 256, use_bias=False))
        layer7 = tf.nn.leaky_relu(tf.layers.dense(layer6, 300, use_bias=False))
        ae_en = tf.layers.dense(layer4, 300)
        ae_de = tf.layers.dense(layer7, 300)
        return ae_en, ae_de



    def MMIN(self):
        mmin_en = []
        mmin_de = []
        x_in = self.enc_new
        for i in range(5):
          #encoder
          layer1 = tf.nn.leaky_relu(tf.layers.dense(x_in, 300, use_bias=False))
          layer2 = tf.nn.leaky_relu(tf.layers.dense(layer1, 256, use_bias=False))
          layer3 = tf.nn.leaky_relu(tf.layers.dense(layer2, 128, use_bias=False))
          layer4 = tf.nn.leaky_relu(tf.layers.dense(layer3, 64, use_bias=False))
          #decoder
          layer5 = tf.nn.leaky_relu(tf.layers.dense(layer4, 128, use_bias=False))
          layer6 = tf.nn.leaky_relu(tf.layers.dense(layer5, 256, use_bias=False))
          layer7 = tf.nn.leaky_relu(tf.layers.dense(layer6, 300, use_bias=False))
          x_in = x_in + layer7
          mmin_en.append(layer4) 
        mmin_en = tf.layers.dense(tf.concat(mmin_en, -1), 300)
        mmin_de = layer7
        return mmin_en, mmin_de

      
    def recovery(self, enc_de):

        de_v = []
        de_a = []
        de_t = []

        for i in range(self.config.batch_size):
            de_v.append(enc_de[i][:100])
            de_a.append(enc_de[i][100:250])
            de_t.append(enc_de[i][250:300])

        re_v = []
        re_a = []
        re_t = []
        for i in range(self.config.batch_size):
            if self.flag[i] == 0:
                tmp_v = de_v[i]
                tmp_a = self.common_a[i]
                tmp_t = self.common_t[i]
            elif self.flag[i] == 1:
                tmp_v = self.common_v[i]
                tmp_a = de_a[i]
                tmp_t = self.common_t[i]
            elif self.flag[i] == 2:
                tmp_v = self.common_v[i]
                tmp_a = self.common_a[i]
                tmp_t = de_t[i]
            else:
                tmp_v = self.common_v[i]
                tmp_a = self.common_a[i]
                tmp_t = self.common_t[i]

            re_v.append(tmp_v)
            re_a.append(tmp_a)
            re_t.append(tmp_t)
        return re_v, re_a, re_t


    def multi_res(self, enc_en, re_v, re_a, re_t, scope):
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
          Wr_va = tf.get_variable('Wr_va', [self.att_dim, 1])
          Wm_va = tf.get_variable('Wm_va', [self.att_dim, self.att_dim])
          Wu_va = tf.get_variable('Wu_va', [self.att_dim, self.att_dim])


          Wr_vt = tf.get_variable('Wr_vt', [self.att_dim, 1])
          Wm_vt = tf.get_variable('Wm_vt', [self.att_dim, self.att_dim])
          Wu_vt = tf.get_variable('Wu_vt', [self.att_dim, self.att_dim])

          Wr_at = tf.get_variable('Wr_at', [self.att_dim, 1])
          Wm_at = tf.get_variable('Wm_at', [self.att_dim, self.att_dim])
          Wu_at = tf.get_variable('Wu_at', [self.att_dim, self.att_dim])

          Wr_vat = tf.get_variable('Wr_vat', [self.att_dim, 1])
          Wm_vat = tf.get_variable('Wm_vat', [self.att_dim, self.att_dim])
          Wu_vat = tf.get_variable('Wu_vat', [self.att_dim, self.att_dim])
          
          Wr_wq = tf.get_variable('Wr_wq', [self.att_dim, 1])
          Wm_wq = tf.get_variable('Wm_wq', [self.att_dim, self.att_dim])
          Wu_wq = tf.get_variable('Wu_wq', [self.att_dim, self.att_dim])


        tmp_en =SigmoidAtt(enc_en, Wr_wq, Wm_wq, Wu_wq)


        #V+A
        cur_va = tf.concat([re_v, re_a], 1)
        with tf.variable_scope('va', reuse=tf.AUTO_REUSE):
          enc_va = multihead_attention(queries=cur_va,
                                   keys=cur_va,
                                   values=cur_va,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          enc_va = ff(enc_va, num_units=[4*self.att_dim, self.att_dim])

        tmp_va = SigmoidAtt(enc_va, Wr_va, Wm_va, Wu_va)

        #V+T
        cur_vt = tf.concat([re_v, re_t], 1)
        with tf.variable_scope('vt', reuse=tf.AUTO_REUSE):
          enc_vt = multihead_attention(queries=cur_vt,
                                   keys=cur_vt,
                                   values=cur_vt,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          enc_vt = ff(enc_vt, num_units=[4*self.att_dim, self.att_dim])


        tmp_vt = SigmoidAtt(enc_vt, Wr_vt, Wm_vt, Wu_vt)


        #A+T
        cur_at = tf.concat([re_a, re_t], 1)
        with tf.variable_scope('at', reuse=tf.AUTO_REUSE):
          enc_at = multihead_attention(queries=cur_at,
                                   keys=cur_at,
                                   values=cur_at,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          enc_at = ff(enc_at, num_units=[4*self.att_dim, self.att_dim])
        tmp_at = SigmoidAtt(enc_at, Wr_at, Wm_at, Wu_at)


        #V+A+T
        cur_vat = tf.concat([re_v, re_a, re_t], 1)

        with tf.variable_scope('vat', reuse=tf.AUTO_REUSE):
          enc_vat = multihead_attention(queries=cur_vat,
                                   keys=cur_vat,
                                   values=cur_vat,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          enc_vat = ff(enc_vat, num_units=[4*self.att_dim, self.att_dim])
        tmp_vat = SigmoidAtt(enc_vat, Wr_vat, Wm_vat, Wu_vat)
        
        multi_tmp = {'en': tmp_en, 'va':tmp_va, 'vt':tmp_vt, 'at':tmp_at, 'vat':tmp_vat}
        
        return multi_tmp


    def ensemble(self, trans_tmp, ae_tmp, mmin_tmp):
        with tf.variable_scope('ensemble', reuse = tf.AUTO_REUSE):
          W_ens = tf.get_variable('W_ens', [self.att_dim, 1])

        res_en = []
        tmp_en = trans_tmp['en']
        res_va = tf.layers.dense(trans_tmp['va'], self.config.class_num)
        res_vt = tf.layers.dense(trans_tmp['vt'], self.config.class_num)
        res_at = tf.layers.dense(trans_tmp['at'], self.config.class_num)
        res_vat = tf.layers.dense(trans_tmp['vat'], self.config.class_num)
        

        ae_va = tf.layers.dense(ae_tmp['va'], self.config.class_num)
        ae_vt = tf.layers.dense(ae_tmp['vt'], self.config.class_num)
        ae_at = tf.layers.dense(ae_tmp['at'], self.config.class_num)
        ae_vat = tf.layers.dense(ae_tmp['vat'], self.config.class_num)

        mm_va = tf.layers.dense(mmin_tmp['va'], self.config.class_num)
        mm_vt = tf.layers.dense(mmin_tmp['vt'], self.config.class_num)
        mm_at = tf.layers.dense(mmin_tmp['at'], self.config.class_num)
        mm_vat = tf.layers.dense(mmin_tmp['vat'], self.config.class_num)

         
        #cal wei
        t_va = tf.reduce_max(res_va, -1)
        t_vt = tf.reduce_max(res_vt, -1)
        t_at = tf.reduce_max(res_at, -1)
        wei_tr = tf.nn.softmax(tf.concat([t_va, t_vt, t_at], 0), 0)


        a_va = tf.reduce_max(ae_va, -1)
        a_vt = tf.reduce_max(ae_vt, -1)
        a_at = tf.reduce_max(ae_at, -1)
        wei_ae = tf.nn.softmax(tf.concat([a_va, a_vt, a_at], 0), 0)

        m_v = tf.reduce_max(mm_va, -1)
        m_a = tf.reduce_max(mm_vt, -1)
        m_t = tf.reduce_max(mm_at, -1)
        wei_mm = tf.nn.softmax(tf.concat([m_v, m_a, m_t], 0), 0)




        for i in range(self.config.batch_size):
            if (self.flag[i] == 0 and tf.argmax(res_vat[i]) != tf.argmax(res_at[i])) or (self.flag[i] == 1 and tf.argmax(res_vat[i]) != tf.argmax(res_vt[i])) or (self.flag[i] == 2 and tf.argmax(res_vat[i]) != tf.argmax(res_va[i])):
                tmp_t = tf.multiply(trans_tmp['va'], wei_tr[0][i]) + tf.multiply(trans_tmp['vt'], wei_tr[1][i]) + tf.multiply(trans_tmp['at'], wei_tr[2][i])
                tmp_ae = tf.multiply(ae_tmp['va'], wei_ae[0][i]) + tf.multiply(ae_tmp['vt'], wei_ae[1][i]) + tf.multiply(ae_tmp['at'], wei_ae[2][i])
                tmp_mm = tf.multiply(mmin_tmp['va'], wei_mm[0][i]) + tf.multiply(mmin_tmp['vt'], wei_mm[1][i]) + tf.multiply(mmin_tmp['at'], wei_mm[2][i])
                #tmp_ = (tmp_t + tmp_ae + tmp_mm)/3
                concat_all = tf.transpose(tf.concat([tmp_t, tmp_ae, tmp_mm], 0))
                wei_s = tf.nn.softmax(tf.matmul(concat_all, W_ens), 0)
                tmp_ = tf.matmul(tf.transpose(wei_s), concat_all)
            else:
                tmp_ = tmp_en[i]
            
            res_en.append(tmp_)

        return res_en

    def cal_loss(self, final_res, enc_en, enc_de):

        with tf.variable_scope('loss_weights', reuse = tf.AUTO_REUSE):

          W_l = tf.get_variable('W_l', [self.att_dim, self.config.class_num])
          b_l = tf.get_variable('b_l', [1, self.config.class_num])


        outputs_en = tf.convert_to_tensor(final_res)
        temp_new = outputs_en
        temp_new = tf.layers.dense(temp_new, self.config.att_dim, use_bias=False)

        output_res = tf.add(tf.matmul(temp_new, W_l), b_l)
        ouput_label = tf.one_hot(self.label, self.config.class_num)
        self.prob = tf.nn.softmax(output_res)


        with tf.name_scope('loss'):
          #encode kl loss
          kl = tf.keras.losses.KLDivergence(reduction = losses_utils.ReductionV2.NONE, name = 'kl')
          kl_loss1 = kl(tf.nn.softmax(enc_en, -1), tf.nn.softmax(self.pretrained_output, -1))
          kl_loss2 = kl(tf.nn.softmax(self.pretrained_output, -1), tf.nn.softmax(enc_en, -1))
          self.kl_loss = tf.reduce_sum(tf.reduce_mean(kl_loss1, -1), -1) + tf.reduce_sum(tf.reduce_mean(kl_loss2, -1), -1)


          de_loss1 = kl(tf.nn.softmax(enc_de, -1), tf.nn.softmax(self.enc_new, -1))
          de_loss2 = kl(tf.nn.softmax(self.enc_new, -1), tf.nn.softmax(enc_de, -1))
          self.de_loss = tf.reduce_sum(tf.reduce_mean(de_loss1, -1), -1) + tf.reduce_sum(tf.reduce_mean(de_loss2, -1), -1)
 

          loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = output_res, labels=ouput_label))
          self.loss = loss
          self.l2_loss = tf.contrib.layers.apply_regularization(regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                  weights_list = [W_l, b_l])
          self.total_loss = self.loss + 0.1*self.l2_loss + 0.1*self.kl_loss + 0.1*self.de_loss
          


