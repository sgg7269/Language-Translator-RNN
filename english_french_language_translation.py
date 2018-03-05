
# coding: utf-8

# # Language Translation
# In this project, I create a neural network machine translator by training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.
# ## Get the Data
# Since translating the whole language of English to French will take lots of time to train, I am only using a small portion of the English corpus.

# In[1]:


import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)


# ## Explore the Data
# Play around with view_sentence_range to view different parts of the data.

# In[2]:


view_sentence_range = (0, 10)

import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


# ## Implement Preprocessing Function
# ### Text to Word Ids
# First we need to turn the text into a number so the computer can understand it. In the function `text_to_ids()`, we turn `source_text` and `target_text` from words to ids.  However, its important we also add the `<EOS>` word id at the end of each sentence from `target_text`.  This will help the neural network predict when the sentence should end.

# In[3]:


def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    source_text_to_id = [[source_vocab_to_int[word] for word in line.split()] for line in source_text.split('\n')]
    target_text_to_id = [[target_vocab_to_int[word] for word in line.split()] + [target_vocab_to_int['<EOS>']] for line in target_text.split('\n')]
    
    return (source_text_to_id, target_text_to_id)
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)


# ### Preprocess all the data and save it
# Running the code cell below will preprocess all the data and save it to file.

# In[4]:


helper.preprocess_and_save_data(source_path, target_path, text_to_ids)


# # Check Point
# This is the first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.

# In[5]:


import numpy as np
import helper

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()


# ### Check the Version of TensorFlow and Access to GPU
# This will check to make sure you have the correct version of TensorFlow and access to a GPU

# In[6]:


from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) in [LooseVersion('1.0.0'), LooseVersion('1.0.1')], 'This project requires TensorFlow version 1.0  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ## Build the Neural Network
# Here I build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:
# - `model_inputs`
# - `process_decoding_input`
# - `encoding_layer`
# - `decoding_layer_train`
# - `decoding_layer_infer`
# - `decoding_layer`
# - `seq2seq_model`
# 
# ### Input
# First I implement the `model_inputs()` function to create TF Placeholders for the Neural Network. It should create the following placeholders:
# 
# - Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
# - Targets placeholder with rank 2.
# - Learning rate placeholder with rank 0.
# - Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.
# 
# Return the placeholders in the following the tuple (Input, Targets, Learing Rate, Keep Probability)

# In[7]:


def model_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """
    input = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return(input, targets, learning_rate, keep_prob)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)


# ### Process Decoding Input
# Here I implement `process_decoding_input` using TensorFlow to remove the last word id from each batch in `target_data` and concat the GO ID to the begining of each batch.

# In[8]:


def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for dencoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    ending = tf.strided_slice(target_data, begin = [0, 0], end = [batch_size, -1], strides = [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)
    
    return dec_input

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_decoding_input(process_decoding_input)


# ### Encoding
# Here I implement the `encoding_layer()` function to create an Encoder RNN layer using [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn).

# In[9]:


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """
    enc_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
    enc_drop = tf.contrib.rnn.DropoutWrapper(enc_cell, keep_prob)
    _, rnn_state = tf.nn.dynamic_rnn(cell = enc_drop, inputs = rnn_inputs, dtype=tf.float32)
    
    return rnn_state
    
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)


# ### Decoding - Training
# Here I create training logits using [`tf.contrib.seq2seq.simple_decoder_fn_train()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_train) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder).  Then I apply the `output_fn` to the [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder) outputs.

# In[10]:


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits
    """
    with tf.variable_scope("decoding") as decoding_scope:
        # Training Decoder
        train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, keep_prob)
        #dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
        train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            dec_cell, train_decoder_fn, dec_embed_input, sequence_length, scope=decoding_scope)

        # Apply output function
        train_logits =  output_fn(train_pred)
    
        return train_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)


# ### Decoding - Inference
# Here I create inference logits using [`tf.contrib.seq2seq.simple_decoder_fn_inference()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_inference) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder). 

# In[11]:


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: The maximum allowed time steps to decode
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """
    with tf.variable_scope("decoding", reuse=None) as decoding_scope:
        # Inference Decoder
        infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
            output_fn = output_fn, encoder_state = encoder_state, embeddings = dec_embeddings, start_of_sequence_id = start_of_sequence_id, end_of_sequence_id = end_of_sequence_id, 
            maximum_length = maximum_length, num_decoder_symbols = vocab_size)
        inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(cell = dec_cell, decoder_fn = infer_decoder_fn, scope=decoding_scope)

        return inference_logits
    
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)


# ### Build the Decoding Layer
# Here I implement the `decoding_layer()` funciton to create a Decoder RNN layer.
# 
# - the RNN cell is created for decoding using `rnn_size` and `num_layers`.
# - the output fuction is created using [`lambda`](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) to transform it's input, logits, to class logits.
# - Used my `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)` function to get the training logits.
# - Used my `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size, decoding_scope, output_fn, keep_prob)` function to get the inference logits.

# In[12]:


def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """
    start_of_sequence_id = target_vocab_to_int['<GO>']
    end_of_sequence_id = target_vocab_to_int['<EOS>']
    
    dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
    dec_drop = tf.contrib.rnn.DropoutWrapper(dec_cell, keep_prob)
    
    with tf.variable_scope("decoding", reuse=None) as decoding_scope:
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
        training_logits = decoding_layer_train(encoder_state, dec_drop, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)
    
    with tf.variable_scope("decoding", reuse=True) as decoding_scope:
        inference_logits = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, sequence_length, vocab_size, decoding_scope, output_fn, keep_prob)

    return(training_logits, inference_logits)
    
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)


# ### Build the Neural Network
# Here I apply the functions I implemented above to:
# 
# - Apply embedding to the input data for the encoder.
# - Encode the input using the function `encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob)`.
# - Process target data using the function `process_decoding_input(target_data, target_vocab_to_int, batch_size)`.
# - Apply embedding to the target data for the decoder.
# - Decode the encoded input using the function `decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)`.

# In[13]:


def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """
    embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, enc_embedding_size)
    enc_state = encoding_layer(embed_input, rnn_size, num_layers, keep_prob)
    dec_input = process_decoding_input(target_data, target_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.truncated_normal([target_vocab_size, dec_embedding_size], stddev = 0.01))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    training_logits, inference_logits = decoding_layer(dec_embed_input, dec_embeddings, enc_state, target_vocab_size, sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)
    return training_logits, inference_logits
    
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)


# ## Neural Network Training
# ### Hyperparameters
# Here I tune the parameters as follows:
# 
# - Set `epochs` to the number of epochs.
# - Set `batch_size` to the batch size.
# - Set `rnn_size` to the size of the RNNs.
# - Set `num_layers` to the number of layers.
# - Set `encoding_embedding_size` to the size of the embedding for the encoder.
# - Set `decoding_embedding_size` to the size of the embedding for the decoder.
# - Set `learning_rate` to the learning rate.
# - Set `keep_probability` to the Dropout keep probability

# In[14]:


# Number of Epochs
epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Number of Layers
num_layers = 1
# Embedding Size
encoding_embedding_size = 13
decoding_embedding_size = 13
# Learning Rate
learning_rate = 0.01
# Dropout Keep Probability
keep_probability = 0.7


# ### Build the Graph
# Here I build the graph using the neural network that was just implemented.

# In[15]:


save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_source_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob = model_inputs()
    sequence_length = tf.placeholder_with_default(max_source_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)
    
    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
        encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)

    tf.identity(inference_logits, 'logits')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


# ### Train
# Here we finally train the neural network on the preprocessed data.

# In[16]:


import time

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1]), (0,0)],
            'constant')

    return np.mean(np.equal(target, np.argmax(logits, 2)))

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]

valid_source = helper.pad_sentence_batch(source_int_text[:batch_size])
valid_target = helper.pad_sentence_batch(target_int_text[:batch_size])

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch) in enumerate(
                helper.batch_data(train_source, train_target, batch_size)):
            start_time = time.time()
            
            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})
            
            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})
                
            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(np.array(valid_target), batch_valid_logits)
            end_time = time.time()
            print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
                  .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')


# ### Save Parameters
# Save the `batch_size` and `save_path` parameters for inference.

# In[19]:


# Save parameters for checkpoint
helper.save_params(save_path)


# # Checkpoint

# In[21]:


import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()


# ## Sentence to Sequence
# To feed a sentence into the model for translation, we first need to preprocess it.  I implement the function `sentence_to_seq()` to preprocess new sentences.
# 
# - First I convert the sentence to lowercase
# - Next I convert the words into ids using `vocab_to_int`
# - Finally I convert words not in the vocabulary, to the `<UNK>` word id.

# In[23]:


def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    lower_case_words = [word.lower() for word in sentence.split()]
    
    word_id = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in lower_case_words]
    
    return word_id

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)


# ## Translate
# This will translate any phrase entered in the variable `translate_sentence` from English to French. Since this is a very small corpus you'll need to only pick words from the dataset used in order to get an accurate translation.

# In[24]:


translate_sentence = 'paris is sometimes warm during june , but it is usually hot in july .'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))


# ## Imperfect Translation
# You might notice that some sentences translate better than others.  Since the dataset we're using only has a vocabulary of 227 English words of the thousands that are actually used in every day life, you're only going to see good results using these words.
# 
# The next step is I plan to train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  
