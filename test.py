import tensorflow as tf
import RE_CNN
import data_deal
import numpy as np

sentence_length=400
labels=4
lr = 0.001
wordvec_size=200
dist_size=20
batch_size=128
filter_sizes=[3 ,4 , 5]
filter_num=100#out_put channel
dropout_keep_prob=0.6
train_data_path="New_Data/train_from_all.txt"
test_data_path="New_Data/test_from_all.txt"
relationid_path="New_Data/relation.txt"
num_epochs=5
use_wiki_vec=True

data = data_deal.data_prepare(train_data_path=train_data_path,relationid_path=relationid_path,sentence_length=sentence_length,batch_size=batch_size,use_wiki_vec=use_wiki_vec)
data.read_train_data()
vocab_size=data.vocab_size



with tf.variable_scope("model"):
    model= RE_CNN.RE_CNN(labels=labels,vocab_size=vocab_size,filter_sizes=filter_sizes,sentence_length=sentence_length,wordvec_size=wordvec_size,batch_size=batch_size,use_wiki_vec=use_wiki_vec,filter_num=filter_num,dropout_keep_prob=dropout_keep_prob)
saver = tf.train.Saver()

with tf.Session() as sess:
# with sess.as_default():
    loss_one_epochs=[]
    accuracy_one_epochs=[]
    saver.restore(sess, 'save/my-model')

    train_data=data.read_train_data()
    def eval_step(input_x, input_y, entity1_pos, entity2_pos):
        feed_dict = {
            model.input_x: input_x,
            model.input_y: input_y,
            model.entity1_pos: entity1_pos,
            model.entity2_pos: entity2_pos,
            model.dropout_keep_prob: 1
        }
        accuracy, predictions = sess.run([model.accuracy, model.predictions],feed_dict=feed_dict)
        accuracy_one_epochs.append(accuracy)
        return predictions
    
    # model.dropout_keep_prob=1
    for sentences_ids,word_en1_pos,word_en2_pos,relations in data_deal.data_iter_random(train_data=train_data, batch_size=batch_size):
        eval_step(
            input_x = np.array(sentences_ids, dtype=np.int32), 
            input_y= np.array(relations), 
            entity1_pos=np.array(word_en1_pos, dtype=np.int32), 
            entity2_pos=np.array(word_en1_pos, dtype=np.int32),
            )
    accuracy_one_epochs_total=0
    for l in accuracy_one_epochs:
                accuracy_one_epochs_total+=l
    print("acc ",(accuracy_one_epochs_total/len(accuracy_one_epochs)))
