import tensorflow as tf
import RE_CNN
import data_deal
import numpy as np

sentence_length=40
labels=12
lr = 0.0001
wordvec_size=200
dist_size=20
batch_size=10
filter_sizes=[3, 5, 7 ,9]
filter_num=200#out_put channel
dropout_keep_prob=0.8
train_data_path="test.txt"
relationid_path="relation2id.txt"
is_training=False
num_epochs=40
use_wiki_vec=True
# sentence_length=30

data = data_deal.data_prepare(train_data_path=train_data_path,relationid_path=relationid_path,sentence_length=sentence_length,batch_size=batch_size,use_wiki_vec=use_wiki_vec)
vocab_size=data.vocab_size
train_data=data.read_train_data()
# print(train_data)
# with tf.Graph().as_default():

with tf.Session() as sess:
# with sess.as_default():
    loss_one_epochs=[]
    accuracy_one_epochs=[]
    with tf.variable_scope("model"):
        model= RE_CNN.RE_CNN(vocab_size=vocab_size,filter_sizes=filter_sizes,is_training=is_training,sentence_length=sentence_length,wordvec_size=wordvec_size,batch_size=batch_size,use_wiki_vec=use_wiki_vec)
    names_to_vars = {v.op.name: v for v in tf.global_variables()}
    saver = tf.train.Saver(names_to_vars)
    # saver =  tf.train.import_meta_graph('save/my-model.meta')
    saver.restore(sess, 'save/my-model')
    for name in names_to_vars:
        print(name)