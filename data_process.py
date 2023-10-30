import torch
# import nltk
import random
import numpy as np
# read one text file, Concatenate all lines in the training set into one long sequence, 
# and tokenize the sequence into words, replacing \n as <eos> 

def read_file(file_path):
    """
    file_path is the path of the file to be read
    output is a list of lines
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def concat_lines(lines, shuffle = False):
    """
    lines is a list of lines
    output is a string of all lines concatenated
    replace \n as <eos>
    """
    if shuffle == True:
        random.shuffle(lines)
    return ' '.join(lines).replace('\n', '<eos>')

def tokenize(line):
    """
    line is one sequence of words
    output is a list of words
    it seems that given texts are well preprocessed
    """
    return line.split()

def read_and_tokenize(file_path):
    """
    file_path is the path of the file to be read
    output is a list of words
    """
    lines = read_file(file_path)
    line = concat_lines(lines, shuffle= True)
    line = tokenize(line)
    return line, len(line)

# Generate batches with shape (batch size, sequence length + 1). There are two ways of batching and we will compare them in subsequent problems.
# Option 1: Continuous Batching. Split the long sequence into batch size equal-length sequences
# (these split sequences are also fairly long sequences). For each split sequence, cut them into con- tinuous parts. Batch k is composed of the kth part of all the sequences. Note that the samples in adjacent batches should be continuous.
# For example, assume the ith sequence is “I eat a banana and an apple everyday including today.”, and sequence length = 3. The ith sample in first two batches should be “I eat a banana” and “banana and an apple” (note the shifting here). With this method, it is suggested to shuffle the articles before concatenation in step 1 between epochs.
# Option 2: Shuffled Batching. Split the long sequence into sequences with length sequence length+ 1. Each batch consists of batch size different sequences. Note that adjacent batches might not be continuous, and you should shuffle them.


def get_vocab(path = './vocab.txt'):
    """
    output is a list of words
    """
    
    with open(path, 'r') as f:
        lines = f.readlines()
    print('length of vocab is {}'.format(len(lines)))
    return [line.strip() for line in lines]

def get_vocab_embed_dict(path = './glove_local.txt'):
    """
    output is a dictionary of words in vocab and their embeddings
    """
    embeddings_dict = {}
    with open(path , 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


class DataLoader():
    def __init__(self, file_path: str  ,  type = 'continuous' , batch_size = 20, seq_len = 35, device = 'cpu'):

        assert type in ['continuous', 'shuffle']
        
        self.filepath = file_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.type = type
        
        self.vocab = get_vocab()
        # self.vocab_embed_dict = get_vocab_embed_dict()
        self.vocab_size = len(self.vocab)
        # self.vocab_embed_dim  = len(self.vocab_embed_dict['the'])
        self.vocab_dict = {word: i for i, word in enumerate(self.vocab)}
        self.device = device

        # self.first_epoch = True
        self.start_epoch()


    def start_epoch(self):
        self.tokens, self.tokens_size = read_and_tokenize(self.filepath)
        if self.type == 'continuous':
            
            self.each_split_len, self.split_remain = self.tokens_size // self.batch_size, self.tokens_size % self.batch_size  
            self.num_batches = (self.each_split_len - 1)// self.seq_len  
            # self.input_data = np.zeros(shape= (self.num_batches, self.batch_size, self.seq_len+1, self.vocab_embed_dim), dtype = np.float32)
            self.target_data = np.zeros(shape= (self.num_batches, self.batch_size, self.seq_len+1), dtype = np.int64)
            
            for j in range(self.batch_size):
                for i in range(self.num_batches): 
                    start = i * self.seq_len + j * self.each_split_len 
                    start += j if j < self.split_remain else self.split_remain
                    token = self.tokens[start: start + self.seq_len+ 2]
                    for k in range(self.seq_len+1):
                        # self.input_data[i,j,k] = self.vocab_embed_dict[token[k]] 
                        self.target_data[i,j,k] = self.vocab_dict[token[k]]

        else: # self.type == 'shuffle'
                self.num_data = (self.tokens_size - 1) // self.seq_len
                self.num_batches = self.num_data // self.batch_size
                # self.input_data = np.zeros(shape= (self.num_batches, self.batch_size, self.seq_len+1, self.vocab_embed_dim), dtype = np.float32)
                self.target_data = np.zeros(shape= (self.num_batches, self.batch_size, self.seq_len+1), dtype = np.int64)
                
                order = np.zeros(shape= (self.num_batches, self.batch_size), dtype= object)
                for i in range(self.num_batches):
                    for j in range(self.batch_size):
                        order[i, j] = (i, j)
                shape = order.shape
                order = order.reshape(-1)
                np.random.shuffle(order)
                order = order.reshape(shape)
                for i in range(self.num_batches):
                    for j in range(self.batch_size):
                        start = j*self.seq_len + i*self.batch_size*self.seq_len
                        token = self.tokens[start: start + self.seq_len+ 2]
                        for k in range(self.seq_len+1):
                            # self.input_data[i,j,k] = self.vocab_embed_dict[token[k]]
                            self.target_data[*order[i, j],k] = self.vocab_dict[token[k]]
                            
                            
        return
    
    def __iter__(self):
        self.start_epoch()
        self.curr_batch = 0
        return self
    
    def __next__(self):
        if self.curr_batch < self.num_batches:
            # input, target = self.input_data[self.curr_batch, :, 0:self.seq_len ,:], self.target_data[self.curr_batch, :, 1:self.seq_len+1]
            input, target = self.target_data[self.curr_batch, :, 0:self.seq_len], self.target_data[self.curr_batch, :, 1:self.seq_len+1]
            self.curr_batch += 1
            return torch.from_numpy(input), torch.from_numpy(target).reshape(self.batch_size * self.seq_len)
            # input, target
        else:
            raise StopIteration 
    
    def get_len(self):
        return self.num_batches

if '__main__' == __name__:
    # tokens, _ = read_and_tokenize('../penn-treebank/ptb.train.txt')
    # print('length of tokens is {}'.format(len(tokens)))
    import time
    curr =  time.time()
    a = DataLoader('./test_loader.txt',  type = 'shuffle', batch_size = 2, seq_len = 7)
    # print('time used is {}'.format(time.time() - curr))
    k = [0,1,]
    for i, (x, y) in enumerate(a):
        if i in k:
            print(x)
            print(y)
    # ran
    ...
