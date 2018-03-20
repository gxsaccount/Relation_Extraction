# test
import numpy as np
import data_deal
import RE_CNN
import tensorflow as tf
from sklearn.metrics import classification_report

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
num_epochs=100
use_wiki_vec=True

data = data_deal.data_prepare(train_data_path=train_data_path,relationid_path=relationid_path,sentence_length=sentence_length,batch_size=batch_size,use_wiki_vec=use_wiki_vec)
data.read_train_data()
vocab_size=data.vocab_size
test_data=data_deal.data_prepare(train_data_path=test_data_path,relationid_path=relationid_path,sentence_length=sentence_length,batch_size=batch_size,use_wiki_vec=use_wiki_vec)
test_data.read_train_data()
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        loss_one_epochs=[]
        accuracy_one_epochs=[]
        accuracy_one_epochs_test=[]

        y_p=[]
        y_a=[]


        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model= RE_CNN.RE_CNN(labels=labels,vocab_size=vocab_size,filter_sizes=filter_sizes,sentence_length=sentence_length,wordvec_size=wordvec_size,batch_size=batch_size,use_wiki_vec=use_wiki_vec,filter_num=filter_num,dropout_keep_prob=dropout_keep_prob)
        optimizer = tf.train.AdamOptimizer(model.lr)
        trian_op = optimizer.minimize(model.loss)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        def train_step(input_x, input_y, entity1_pos, entity2_pos,lexical_ids,dropout_keep_prob):
            feed_dict = {
                model.input_x: input_x,
                model.input_y: input_y,
                model.entity1_pos: entity1_pos,
                model.entity2_pos: entity2_pos,
                model.lexical_ids:lexical_ids,
                model.dropout_keep_prob: dropout_keep_prob
            }
            _, loss, accuracy = sess.run(
                [trian_op, model.loss, model.accuracy],feed_dict=feed_dict)
            loss_one_epochs.append(loss)
            accuracy_one_epochs.append(accuracy)
            # print("loss {:g}, acc {:g}".format(loss,accuracy))
        

       
        def eval_step(input_x, input_y, entity1_pos, entity2_pos,lexical_ids):
            feed_dict = {
                model.input_x: input_x,
                model.input_y: input_y,
                model.entity1_pos: entity1_pos,
                model.entity2_pos: entity2_pos,
                model.lexical_ids:lexical_ids,
                model.dropout_keep_prob: 1
            }
            accuracy1, predictions = sess.run([model.accuracy, model.predictions],feed_dict=feed_dict)
            accuracy_one_epochs_test.append(accuracy1)
            
            # y_p=np.concatenate(y_p,predictions)
            # y_a=np.concatenate(y_a,np.argmax(input_y))
            # print(predictions["classes"])
            y_a.extend(list(np.argmax(input_y,1)))
            y_p.extend(list(predictions["classes"]))
            # print(np.argmax(input_y, 1))
            # print("acc {:g}".format(accuracy1))
            return predictions
        
        print("------------------------------开始训练----------------------")
        train_data=data.read_train_data()
        train_data2=test_data.read_train_data()
        for i in range(num_epochs):
            for sentences_ids,word_en1_pos,word_en2_pos,relations,lexical_ids in data_deal.data_iter_random(train_data=train_data2, batch_size=batch_size):
                # print(lexical_ids)
                eval_step(
                    input_x = np.array(sentences_ids, dtype=np.int32), #单词向量
                    input_y= np.array(relations, dtype=np.int32), #关系
                    entity1_pos=np.array(word_en1_pos, dtype=np.int32), #实体1位置
                    entity2_pos=np.array(word_en1_pos, dtype=np.int32),  #实体2位置
                    lexical_ids=np.array(lexical_ids, dtype=np.int32)
                )
                
            accuracy_one_epochs_total_test=0
            for l in accuracy_one_epochs_test:
                accuracy_one_epochs_total_test+=l
            print("test acc ",(accuracy_one_epochs_total_test/len(accuracy_one_epochs_test)))
            print(classification_report(y_a,y_p))
            del y_a[:]
            del y_p[:]
            del accuracy_one_epochs_test[:]


            for sentences_ids,word_en1_pos,word_en2_pos,relations,lexical_ids in data_deal.data_iter_random(train_data=train_data, batch_size=batch_size):
                train_step(
                    input_x = np.array(sentences_ids, dtype=np.int32), 
                    input_y= np.array(relations, dtype=np.int32), 
                    entity1_pos=np.array(word_en1_pos, dtype=np.int32), 
                    entity2_pos=np.array(word_en1_pos, dtype=np.int32),
                    lexical_ids=np.array(lexical_ids, dtype=np.int32),
                    dropout_keep_prob=dropout_keep_prob
                    )
            loss_one_epochs_total=0
            for l in loss_one_epochs:
                loss_one_epochs_total+=l
            
            accuracy_one_epochs_total=0
            for l in accuracy_one_epochs:
                accuracy_one_epochs_total+=l
            
            print("{:g} loss {:g}, acc {:g}".format(i,loss_one_epochs_total/len(loss_one_epochs),accuracy_one_epochs_total/len(accuracy_one_epochs)))
            del loss_one_epochs[:]
            del accuracy_one_epochs[:]

            
        # print(model.lexical_level_feature)
        saver.save(sess, 'save/my-model')
        print("save model fin.")



# for sentences_ids,word_en1_pos,word_en2_pos,relations in data_deal.data_iter_random(train_data=train_data2, batch_size=batch_size):
#     eval_step(
#         input_x = np.array(sentences_ids, dtype=np.int32), 
#         input_y= np.array(relations), 
#         entity1_pos=np.array(word_en1_pos, dtype=np.int32), 
#         entity2_pos=np.array(word_en1_pos, dtype=np.int32),
#         dropout_keep_prob=1
#         )
# accuracy_one_epochs_total_test=0
# for l in accuracy_one_epochs_test:
#     accuracy_one_epochs_total_test+=l
# print("acc ",(accuracy_one_epochs_total_test/len(accuracy_one_epochs_test)))