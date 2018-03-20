import numpy as np 
import tensorflow as tf
import pickle
import sklearn as sk
class RE_CNN(object):
    def __init__(self,sentence_length=30,labels=12, lr = 1e-3,vocab_size=2048000, wordvec_size=100,dist_size=10,batch_size=50,
    filter_sizes=[3, 4, 5 ], filter_num=200,dropout_keep_prob=0.8,use_wiki_vec=False,is_training=True):
        # init Domain
        self.sentence_length=sentence_length
        input_x= tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_x")
        self.input_x = input_x
        input_y = tf.placeholder(tf.int32, shape=[batch_size, labels], name="input_y")
        self.input_y = input_y
        entity1_pos = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="entity1_pos")
        self.entity1_pos = entity1_pos

        entity2_pos = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="entity2_pos")
        self.entity2_pos = entity2_pos

        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob=dropout_keep_prob

        lexical_ids=tf.placeholder(tf.int32,shape=[batch_size,6],name="lexical_ids")
        self.lexical_ids=lexical_ids

        # self.dropout_keep_prob = dropout_keep_prob
        self.lr =lr
        if not is_training:
            dropout_keep_prob=1.0
        # get [word_vecs,dist1_vec,dist2_vec]
        with tf.name_scope("word_embedding_layer"):
            # word_vecs
            if use_wiki_vec==False:
                word_vecs = tf.Variable(tf.random_normal(shape=[vocab_size,wordvec_size]),dtype=tf.float32,name="word_table")
                input_word_vecs=tf.nn.embedding_lookup(word_vecs,input_x)
            else:
                with open('embedding_weights.pkl', 'rb') as ewf:
                    word_vecs = tf.constant(pickle.load(ewf),dtype=tf.float32)
                    input_word_vecs = tf.nn.embedding_lookup(word_vecs,input_x)
                    
                
  
            # print("input_word_vecs",input_word_vecs)
            
            # dist1_vec
            entity1_pos_postive = entity1_pos+(sentence_length-1)
            dist1_vecs = tf.Variable(tf.random_normal(shape=[2*sentence_length-1,dist_size]),name="input_dist1_table")
            input_dist1_vecs =tf.nn.embedding_lookup(dist1_vecs,entity1_pos_postive)
            # dist2_vec
            entity2_pos_postive = entity2_pos+(sentence_length-1)
            dist2_vecs = tf.Variable(tf.random_normal(shape=[2*sentence_length-1,dist_size]),name="input_dist2_table")
            input_dist2_vecs =tf.nn.embedding_lookup(dist2_vecs,entity2_pos_postive)

            
            print("lexical_ids",lexical_ids.shape)
            # lexical_vecs=tf.expand_dims(input_word_vecs)
            lexical_vec=tf.reshape(tf.nn.embedding_lookup(word_vecs,lexical_ids),[batch_size,6*wordvec_size])
            print("lexical_vec",lexical_vec.shape)


            # CNN input,concat with sentence_length [[batch_size, sentence_length, word/dist_size]]
            # add a dim as channl to fit CNN inputs[batch,weight,hight,channl]
            sentence_vec =  tf.expand_dims(tf.concat([input_word_vecs,input_dist1_vecs,input_dist2_vecs],2),-1)
            print("sentence_vec",sentence_vec.shape)
        # convolutional layer
        pooled_outputs =[]
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # [filter_height, filter_width, in_channels, out_channels]
                # print(wordvec_size)
                # print(dist_size)
                filter_shape = [filter_size , wordvec_size+2*dist_size , 1 , filter_num]
                # print("filter_shape",filter_shape)

                weight=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='weight')
                bias=tf.Variable(tf.constant(0.1,shape=[filter_num]),name='bias')

                # CNN
                conv=tf.nn.conv2d(sentence_vec,filter=weight,strides=[1,1,1,1],padding="VALID",name="conv")
                # print("conv",conv)
                # activate 
                relu_output=tf.nn.relu(tf.nn.bias_add(conv,bias),name="relu")
                # print("relu_output",relu_output)
                # pooling
                pool_output=tf.nn.max_pool(
                    relu_output,ksize=[1, sentence_length - filter_size+1,1,1],strides=[1, 1, 1, 1],padding="VALID",name="pool")
                # print("pool_output",pool_output)
                pooled_outputs.append(pool_output)
            # ????2or3?
        combine_pooled=tf.concat(pooled_outputs,3)
        # print("combine_pooled",combine_pooled.shape)
        combine_pooled_flatten = tf.reshape(combine_pooled, [-1, filter_num * len(filter_sizes)])
        # print("combine_pooled_flatten",combine_pooled_flatten.shape)


        with tf.name_scope("concat_lexical"):
            combine_pooled_flatten=tf.concat([combine_pooled_flatten,lexical_vec],1)

        with tf.name_scope("dropout"):
            print("dropout_keep_prob",dropout_keep_prob)
            feature_dropouted=tf.nn.dropout(combine_pooled_flatten, keep_prob=dropout_keep_prob)
            # print("feature_dropouted",feature_dropouted.shape)




    
        # full connected 
        with tf.name_scope("dense"):
            # ???
            dense = tf.layers.dense(
                inputs=feature_dropouted,
                units=256,
                activation=tf.nn.relu
            )

            dense1 = tf.layers.dense(
                inputs=dense,
                units=256,
                activation=tf.nn.relu
            )

            dense2 = tf.layers.dense(
                inputs=dense1,
                units=256,
                activation=tf.nn.relu
            )
            dropout=tf.layers.dropout(
                inputs=dense2,
                rate=1-dropout_keep_prob,
                training=True
            )
        with tf.name_scope("softmax"):
            logits = tf.layers.dense(
                inputs=dropout,
                units=labels,
            )
            predictions = {
                "classes": tf.argmax(input=logits, axis=1),
                "probalities": tf.nn.softmax(logits, name="softmax_tensor")
            }
            self.predictions = predictions
        
        with tf.name_scope("loss"):
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels= input_y)
            loss = tf.reduce_mean(entropy)
            self.loss=loss
        
        with tf.name_scope("accuracy"):
            x=predictions["classes"]
            y=tf.argmax(input_y, 1)
            correct = tf.equal(x,y )#TP
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
            self.accuracy = accuracy 

            
        
            # for label in range(labels):
            #     TP=tf.count_nonzero(xFP)
            #     TF=tf.count_nonzero(x)
            #     FN=tf.count_nonzero(x)
            #     FP=tf.count_nonzero(x)
            #     for i in range(len(x)):
            #         if x[i]==label and y[i]==label:
            #             TP+=1
            #         elif x[i]!=label and y[i]!=label:
            #             FN+=1
            #         elif x[i]==label and y[i]!=label:
            #             FP+=1
            #         elif x[i]!=label and y[i]==label:
            #             TF+=1
            #     P=TP/(TP+FP)
            #     R=TP/(TP+FN)
            #     F=2*TP/(2*TP+FP+FN)
            #     self.prfs[label]=[P,R,F]
            
            