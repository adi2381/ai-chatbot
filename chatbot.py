# -*- coding: utf-8 -*-
"""
Created on Thu Oct 7 23:48:33 2019

@author: Aditya
"""


########### Building a ChatBot with Deep NLP ###########
 

# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time
 


### Phase 1: Data Preprocessing ###
 


# Importing the dataset
'''
Adding encoding = 'utf-8', errors = 'ignore' to open file commands below are to make sure that any files
such as dataset files, csv etc added to the program are encoded properly by which we mean that they are
set to a language that can be understood by the machine since we have various file formats
'''
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
 
# Creating a dictionary that maps each line with its id
# _line is a temp variable created that is used only for the loop and cannot be used later
id_to_line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id_to_line[_line[0]] = _line[4]
 
# Creating a list of all of the conversations
'''
We use an indexing of [:-1] in conversations because -1 is the last row of 
conversations and by taking : in [:-1], we take an index ranging from beginning
to the last row but since the upper bound is excluded, so, it will also exclude
the last row which is the empty row

To simplify conversations dataset, we used split() to remove brackets, spaces and quotes
for the purpose of which we introduced another temp variable _conversation
'''
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
 
# Getting questions and answers seperately
'''
We will get questions(which are input) and answers(which are target) seperately
before clean up of questions and answers to simplify the dataset as much as possible

conversation[i] refers to the index of the first element, so index for answer
we want to append should be the next index, so, it will be [i+1]
'''
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id_to_line[conversation[i]])
        answers.append(id_to_line[conversation[i+1]])
 
# Simplifying and cleaning the text using Regular Expressions
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text
 
# Cleaning questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
 
# Cleaning answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
 
# Filtering out the questions and answers that are too short or too long
short_questions = []
short_answers = []
i = 0
for question in clean_questions:
    if 2 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(clean_answers[i])
    i += 1
clean_questions = []
clean_answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        clean_answers.append(answer)
        clean_questions.append(short_questions[i])
    i += 1
 
# Creating a dictionary that maps each word to its number of occurrences
'''
We are shifting the sequences of questions and answers by 1, therefore the
previous answer beccomes the next question after the shift due to which we
get an overlap in conversation when it goes back and forth for several lines
'''
word_to_count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word_to_count:
            word_to_count[word] = 1
        else:
            word_to_count[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word_to_count:
            word_to_count[word] = 1
        else:
            word_to_count[word] += 1
 
# Creating two dictionaries that map the words in the questions and the answers to a unique integer
'''
We created two different dictionaries so we can apply different thresholds in order to
filter out non-frequent words in both the dictionaries. Also, both the dictionaries
ended up being identical because answers to previous questions become next questions
to the next answers.
'''
threshold_questions = 15
questions_words_to_int = {}
word_number = 0
for word, count in word_to_count.items():
    if count >= threshold_questions:
        questions_words_to_int[word] = word_number
        word_number += 1
        
threshold_answers = 15
answers_words_to_int = {}
word_number = 0
for word, count in word_to_count.items():
    if count >= threshold_answers:
        answers_words_to_int[word] = word_number
        word_number += 1
 
# Adding the last tokens to above two dictionaries
''' 
<SOS> -> Appended to Start of sequence
<EOS> -> Appended to End of sequence
<OUT> -> Filtering words out
<PAD> -> Replaces empty cells using <PAD> token
'''
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questions_words_to_int[token] = len(questions_words_to_int) + 1
for token in tokens:
    answers_words_to_int[token] = len(answers_words_to_int) + 1
 
# Creating the inverse dictionary of the answers_words_to_int dictionary
'''
We create an inverse dictionary because of the need to do inverse mapping
from int to answer_words  in implementation of seq2seq model, also, we need
to do this only for answer_words_to_int dictionary
'''
answers_int_to_words = {w_i: w for w, w_i in answers_words_to_int.items()}
 
# Adding the <EOS> token to the end of every answer
'''
We only do this for answers because the first element required by the
decoder is the <SOS>  token and last element is <EOS> token
'''
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
 
# Translating all the questions and the answers into int & replacing all the words that were filtered out by <OUT> token
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questions_words_to_int:
            ints.append(questions_words_to_int['<OUT>'])
        else:
            ints.append(questions_words_to_int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answers_words_to_int:
            ints.append(answers_words_to_int['<OUT>'])
        else:
            ints.append(answers_words_to_int[word])
    answers_into_int.append(ints)
 
# Sorting questions and answers by the length of questions
'''
Sorting by length helps us to reduce loss through reducing the use of <PAD> tokens
used to replace empty cells and thus helps to speed up the training
'''
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
 
    
    
### Phase 2: Building SEQ2SEQ Model ###
 
    
    
# Creating placeholders for the inputs and the targets
'''
The tf.placeholder accepts 3 parameters (type of data, dimensions of input data, name for input)

Dimensions of the input data must be 2 dimensional(in tf.placeholder) because neural networks
can only accept inputs that are in batches, as opposed to single input. Thus, we add 1 dimension
corresponding to the batch.

keep_prob is a hyperparameter used to control the dropout rate. Dropout rate is the rate of the
neurons you choose to overwrite during the one iteration of the training.
'''
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob
 
# Preprocessing the targets
'''
We have to delete last column of answer dict before adding <SOS> to process target fuctions to
preserve the max sequence length since after that we do a concat to add <SOS> token at the
beginning of the sequence, thus, we must remove the last token befor so that the sequence
length doesn't go over the max sequence length

Need for preprocessing? -> Decoder only accepts a certain format of targets(targets must be in batches) 
so preprocessing is necessary 

# Explanation: 
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1])
    : [0,0] > is the upper left side of tensor (from where you want to begin the extraction)
    : [batch_size, -1] > taking all the lines except the last column (end, up to where we want to make the extraction)
    : Last argument tells by how many cells we want to slice when doing the extraction and 
    since we want to get all the lines except last column, we will use a slice of [1,1]
'''
def preprocess_targets(targets, word_to_int, batch_size):
    left_side = tf.fill([batch_size, 1], word_to_int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets
 
# Creating the Encoder RNN
'''
LSTM Dropout is a technique used for regularization in neural nets
and it helps in learning and preventing overfitting with the data.

Overfitting > happens when a model learns the detail and noise in the training data 
to the extent that it negatively impacts the performance of the model on new data

Encoder_cell >  the cell inside the encoder RNN that contains the stacked LSTM layers on which
                 we apply dropout to improve accuracy
Encoder_state > output returned by the encoder RNN, right after the last fully connected layer

# Explanation:
    def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length)
    : rnn_inputs = corresponds to the model inputs
    : rnn_size = number of input tensors in encoder rnn layer
    : num_layers = number of layers
    : keep_prob = used to apply dropout to improve accuracy of model
    : sequence_length = length of list of each question in the batch
'''
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state
 
# Decoding the training set
'''
> We need the decode training set to decode the encoded questions and answers of the training set (the second part of the Seq2Seq model)

> This function thus, decodes the observations of the training set and also returns output of the decoder at the same time which
was for the observation of the training set that is some observation going bak in the neural network to update the weights 
and thus, update the ability of the chatbot to talk like a human.

# Explanation 
    def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    : encoder_state = decoder gets encoder_state as input to to decoding
    : decoder_cell = cell in the RNN of the decoder. It contains stacked LSTM layers with dropout applied
    : decoder_embedded_input = embedding means mapping from discreet objects, such as words, to vectors of real numbers
    : sequence_length = length of list of each question in the batch
    : decoding_scope = similar to variable scope, decoding_scope is an object of the var_scope. It is an advanced Data structure
                        that wraps our tensorflow variables
    : output_function = used to return output of the decoding
    : keep_prob = used to apply dropout to improve accuracy of model
    : batch_size = size of batch (we work with batches in seq2seq)

The purpose of the embeddings matrix is to compute more efficiently the embedding input.
Basically, you multiply your vector of inputs by the embeddings matrix to get your embedded inputs. 
'''
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    '''
    attention_states; First, we need to get the attention states.
    To get them, we will have to initialize them as 3-D matrices containing only 0’s.
    Batch_size is the number of lines, number of columns is going to be 1 and axis is 
    going to be decoder_Cell.output_size
    '''
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    '''
    attention keys -> keys to be compared with target states
    attention values -> values that we use to construct context vectors, it is returned by encoder and it should be used by decoder as first element of decoding
    attention score function -> used to compute similarity between keys and the target state
    attention_construct_function -> used to build attention states

    '''
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
 
# Decoding the test/validation set
'''
-> we need the decode test set to decode the encoded questions answers of either the validation set, or simply new predictions that are not used anyway in the training.
-> Now, we make same function as above but for new type of observations, these observations are of test set and validation set and
    won't be used for training.
-> This function is used to predict the oversation of questions we ask the chatbot in the test phase as well as validate them
    Validation set is the set we make during training
-> Key difference between this and previous function is that previous function used ".attention_decoder_fn_train" and in this we use ".attention_decoder_fn_inference",
    inference term in this means to deduce logic. The chatbot thus deduce answers logically to answers the questions we ask. So, it understands its own logic and on new
    observations it will infer for the questions asked.
-> This function returns the Test_predictions
'''
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
 
# Creating the Decoder RNN
'''
-> the Lambda function creates fully connected layers which will get features from stacked LSTM and return the final scores

# Explanation : weights = tf.truncated_normal_initializer(stddev = 0.1)
             -> We need to initialize some weights which will be associated with the neurons of the
                fully connected layer in our decoder. With above line of code, we have our fully connected weights initialized.
'''
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
 
# Building the seq2seq model
'''
-> This function returns the training predictions and the test predictions but we want them to be returned at the same time. 
    Thus, we are going to assemble encoder that returns encoder state and decoder that returns training and test predictions.
-> But we also need encoder because in order for decoder to return test and training predictions, it needs to take encoder state returned by 
    encoder and that is why we need those 2 networks in this function.
'''
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questions_words_to_int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questions_words_to_int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questions_words_to_int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
 
    

### Phase 3: Training the SEQ2SEQ Model ###



# Setting the Hyperparameters
'''
> Epochs : an epoch is a whole process of getting the batches of inputs into the neural network and then for propagating them inside the encoders 
    to get the encoder state and for propagating the encoder states with the targets inside the decoder RNN to get the final output. First, final output scores 
    then final answers predicted by chatbot and then back propagating the loss generated by the output and the target back into the Neural network and updating the weight 
    of the better ability of the chatbot to work like a human. An epoch is basically one whole iteration of the training.
> Num_layers : number of layers you have in encoder and decoder RNN
> Encoding_embedding_size : it Is the number of columns in embeddings matrix that is the number of columns you want to have for 
    embeddings value where in matrix each line corresponds to each token in the whole question of the corpus
> Learning_rate : it must be not be too high or too low. If it is too high the model will learn too fast and will not therefore not learn how to speak properly and 
    if it is slow it will take a long for model to learn properly.
> Learning rate decay :  it means by what percentage the learning rate is reduced over the iterations of the training because we want to start with some learning rate 
    but we want to apply decay over the iterations of the training to reduce the learning rate so it can learn more in depth the logic of human conversations and in our case, 
    or general the correlations found in the dataset.
> keep_prob : used to apply dropout to improve accuracy of model
'''
epochs = 15
batch_size = 32
rnn_size = 1024
num_layers = 3
encoding_embedding_size = 1024
decoding_embedding_size = 1024
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5
 
# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()
 
# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()
 
# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')
 
# Getting the shape of the inputs tensor
'''
What is shape of a tensor?
All values in a tensor hold identical data type with a known (or partially known) shape. 
The shape of the data is the dimensionality of the matrix or array.
'''
input_shape = tf.shape(inputs)
 
# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answers_words_to_int),
                                                       len(questions_words_to_int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questions_words_to_int)
 
# Setting up the Loss Error, the Optimizer and Gradient Clipping
'''
We are using two learning rates:
    : The first learning rate in model inputs was used in the function to create placeholders for inputs and targets since we are working with TensorFlow (essentially going from arrays to tensors for TF) 
        we have to create the placeholder for it. When we are using the learning rate for training, we are preparing the optimizer but we are using the learning rate that we already prepared in the hyper parameters(0.001)
    : Gradient Clipping > it is a technique that will cap the gradients in the graph between a minimum value and maximum value and that is to avoid exploding or vanishing gradients issue.

'''
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
 
# Padding the sequences with the <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
 
# Splitting the data into batches of questions and answers
'''
    We need the start_index to set the first index of the question we are adding in the batch because we are dealing with a specific batch. 
    So, it's the first question / answer we are adding to the batch. To get it we are using batch index * batch size due to starting at index 0. 
    The first question added will have the start index of 0, overall in the loop of split into batches it's being used for that special indexing.
'''
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questions_words_to_int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answers_words_to_int))
        yield padded_questions_in_batch, padded_answers_in_batch
 
# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]
 
# Training
'''
> Total_training_loss_error: will be used to compute training losses on 100 batches
> List_validation_loss_error: we have to make a list, we have to use the early stepping technique which consists of checking if we reached a loss that is 
    below the minimum of all the losses we got and all the losses we get, we put it in this list.
> Early_stopping_check : corresponds to the number of checks each time there is no improvement of validation loss, so each time we don’t reduce the validation loss, 
    early stepping check is going to be incremented by one and one it reaches a certain number which is going to be our next variable -> early_stopping_stop, we will stop the training.
    we choose 1000 because we want to make the training last until the last epoch which is 100, but if you want to apply early stepping then use a value of 100.
> Checkpoint : this variable will be used to save the weights which we will be able to load whenever we want to chat with our trained chatbot.
> Session.run(tf.global_variable_initializer()) : used to run a session
> Batch_time = ending_time – starting_time : used to compute the training time of each batch

'''
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I can speak better now!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry, I need to practice more to speak better.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better than this. This is the best I can do.")
        break
print("Over")
 


### Phase 4: Testing The Seq2Seq Model ###


 
# Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)
 
# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
 
# Setting up the chat
while(True):
    question = input("You: ")
    if question == 'goodbye':
        break
    question = convert_string2int(question, questions_words_to_int)
    question = question + [questions_words_to_int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answers_int_to_words[i] == 'i':
            token = ' I'
        elif answers_int_to_words[i] == '<EOS>':
            token = '.'
        elif answers_int_to_words[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answers_int_to_words[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)