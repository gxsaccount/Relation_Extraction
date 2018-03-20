import jieba
import numpy as np
from gensim.models import Word2Vec
import pickle 


# =Word2Vec.load("embedding_weights.pkl")

def get_wiki_vec(word):
    return wiki_word_vec.wv[word]

def one_hot(size):
        result = []
        for i in range(size):
            one_hot_vec = [0] * size
            one_hot_vec[i] = 1
            result.append(one_hot_vec)
        return result

def find_index(x, y):
    flag = 0
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag

def read_relation(relationid_path):
    relation2id={}
    with open(relationid_path) as f:
        while True:
            content = f.readline()
            if content == '':
                break
            content = content.strip().split()
            relation2id[content[0]] = int(content[1])
    return relation2id
class data_prepare(object):
    def __init__(
        self,
        # train_data_path="train.txt",
        # relationid_path="relation2id.txt",
        train_data_path="Data/train.txt",
        relationid_path="Data/relation.txt",
        batch_size=10,
        sentence_length=30,
        use_wiki_vec=False
        ):
        self.use_wiki_vec=use_wiki_vec
        self.train_data_path=train_data_path
        self.relationid_path=relationid_path
        self.batch_size=batch_size
        self.sentence_length=sentence_length
        self.input_x={}
        self.input_y={}
        
        # self.printInfo()
        
    def printInfo(self):
        print(self.use_wiki_vec,
        self.train_data_path,
        self.relationid_path,
        self.batch_size,
        self.sentence_length)

    def read_train_data(self):
        use_wiki_vec=self.use_wiki_vec
        if use_wiki_vec:
            with open('index_dic.pkl', 'rb') as idf:
                wiki_word_to_id=pickle.load(idf)
        sentence_length=self.sentence_length
        # print(self.relationid_path)
        relation2id=read_relation(self.relationid_path)
        print(relation2id)
        wordSet=set()
        sentences=[]
        en1_pos=[]
        en2_pos=[]
        relations=[]
        sentences_ids=[]
        word_en1_pos=[]
        word_en2_pos=[]
        lexical_ids=[]
        one_hot_relations=one_hot(len(relation2id))
        maxposdict=0
        with open(self.train_data_path) as f:
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                if len(content)<3:
                    print(content)
                    continue
                # get entity name
                en1 = content[0]
                en2 = content[1]
                relation = 0
                if content[2] not in relation2id:
                    relation =relation2id['unknown']
                else:
                    relation=relation2id[content[2]]
                sentence_cut=jieba.cut(content[3])
                sentence=list(sentence_cut)
                for word in sentence:
                    if word is not None:
                        wordSet.add(word)
                maxposdict=max(maxposdict,abs(find_index(en1,sentence)-find_index(en2,sentence)))
                en1_pos.append(find_index(en1,sentence))
                en2_pos.append(find_index(en2,sentence))
                sentences.append(sentence)
                relations.append(one_hot_relations[relation])
            id_to_word=list(wordSet)
            word_to_id=dict((char,i)for i,char in enumerate(id_to_word))
            for s in sentences:
                tmp=[]
                count=0
                for word in s:
                    count+=1
                    if use_wiki_vec==False:
                        tmp.append(word_to_id[word])
                    else:
                        if word in wiki_word_to_id.keys():
                            tmp.append(wiki_word_to_id[word])
                        else:
                            tmp.append(0)
                    if count==sentence_length:
                        break
                for _ in range(sentence_length-len(tmp)):
                    tmp.append(0)
                sentences_ids.append(tmp)
                
            for i in range(len(sentences_ids)):
                sentences_id=sentences_ids[i]
                lexical_id=[]
                p=en1_pos[i]
                if p-1>=0:
                    lexical_id.append(sentences_id[p-1])
                else:
                    lexical_id.append(0)
                lexical_id.append(sentences_id[en1_pos[p]])
                if p+1>=0:    
                    lexical_id.append(sentences_id[en1_pos[p+1]])
                else:
                    lexical_id.append(0)

                p=en2_pos[i]
                if p-1>=0:
                    lexical_id.append(sentences_id[p-1])
                else:
                    lexical_id.append(0)
                lexical_id.append(sentences_id[en1_pos[p]])
                if p+1>=0:    
                    lexical_id.append(sentences_id[en1_pos[p+1]])
                else:
                    lexical_id.append(0)
                lexical_ids.append(lexical_id)
            # print(lexical_ids[0])
                

            # id_to_word_test=id_to_word
            train_data=[]
            for i in range(len(sentences_ids)):
                tmp1=[]
                tmp2=[]
                for k in range(sentence_length):
                    tmp1.append(en1_pos[i]-k)
                    tmp2.append(en2_pos[i]-k)
                word_en1_pos.append(tmp1)
                word_en2_pos.append(tmp2)   


            for i in range(len(sentences_ids)):
                tmp=[sentences_ids[i],word_en1_pos[i],word_en2_pos[i],relations[i],lexical_ids[i]]
                if len(tmp)==5:
                    train_data.append(tmp)
            self.vocab_size=len(wordSet)

            count1=0
            count2=0
            for pos in en1_pos:
                if pos==0:
                    count1+=0
            count2=0
            for pos in en2_pos:
                if pos==0:
                    count2+=0
            print("count1",count1," count2",count2," maxposdict",maxposdict)
            return train_data
            

def data_iter_random(train_data,batch_size):
    np.random.shuffle(train_data)
    num_examples = (len(train_data) - 1)    #样本个数
    epoch_size = num_examples // batch_size
    lentemp=train_data[0]
    sentence_length=len(lentemp[0])
    pos_length=len(lentemp[1])
    relation_length=len(lentemp[3])
    for i in range(epoch_size):
        i = i * batch_size
        sentences_ids = []#train_data[0][i: i + batch_size]
        word_en1_pos=[]#train_data[1][i: i + batch_size]
        word_en2_pos=[]#train_data[2][i: i + batch_size]
        relations=[]#train_data[3][i: i + batch_size]
        lexical_ids=[]
        maxLen=min(i+batch_size,len(train_data))
        for j in range(i,maxLen):
            tmp=train_data[j]
            sentences_ids.append(tmp[0])
            word_en1_pos.append(tmp[1])
            word_en2_pos.append(tmp[2])
            # for k in range(len(sentences_ids))
            relations.append(tmp[3])
            lexical_ids.append(tmp[4])
        # if maxLen<i+batch_size:
        #     for j in range(maxLen,i+batch_size):
        #         sentences_ids.append([0]*sentence_length)
        #         word_en1_pos.append([0]*pos_length)
        #         word_en2_pos.append([0]*pos_length)
        #         relations.append([0]*relation_length)
        yield sentences_ids,word_en1_pos,word_en2_pos,relations,lexical_ids


data= data_prepare(use_wiki_vec=True)
# print(data.vocab_size)
# train_data=read_train_data(train_data_path="train.txt",relationid_path="relation2id.txt",sentence_length=30)
# # while True:
#     # print(data_iter_random(train_data,10))
# i=0
# for sentences_ids,word_en1_pos,word_en2_pos,relations in data_iter_random(data.read_train_data(), 10):
#     print(sentences_ids[i])
#     i+=1
#     i%100