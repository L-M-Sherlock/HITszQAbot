import tensorflow as tf

class CNNClassifier(object):
    """
    A CNN model for text classification.
    Network structure: embedding layer > convolution layer > max-pooling > softmax
    """

    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="input_x")
        #self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.int32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.training = tf.compat.v1.placeholder(tf.int32, name="trainable")
        self.WW = []
        self.bb = []
        # Keeping track of l2 regularization loss (optional)
        if self.training == 1:
            self.TRAIN = True
        else:
            self.TRAIN = False
        l2_loss = tf.compat.v1.constant(0.0)

        # Embedding layer
        # TODO: check tf.device('/cpu:0')
        
        with tf.compat.v1.name_scope("embedding"):
            self.W = tf.compat.v1.Variable(
                tf.compat.v1.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")

            self.embdded_chars = tf.compat.v1.nn.embedding_lookup(self.W, self.input_x)
            self.embdded_chars_expanded = tf.compat.v1.expand_dims(self.embdded_chars, -1)
        
        #self.embdded_chars_expanded = tf.expand_dims(self.input_x, -1)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.compat.v1.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="W_conv")
                
                b = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_filters]), name="b_conv")
                #W = tf.get_variable('W_conv_', filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                #b = tf.get_variable('b_conv_', [num_filters], initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.compat.v1.nn.conv2d(
                    self.embdded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                self.WW.append(W)
                self.bb.append(b)
                # Apply nonlinearity
                #batch_normal = tflearn.layers.normalization.batch_normalization(conv, trainable=self.TRAIN)
                h = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.compat.v1.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")
                print('max pool, ', pooled)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.compat.v1.concat(pooled_outputs, 3)
        print('h_pool, ', self.h_pool)
        self.h_pool_flat = tf.compat.v1.reshape(self.h_pool, [-1, num_filters_total])
        print(self.h_pool_flat)

        # Add dropout
        with tf.compat.v1.name_scope("dropout"):
            self.h_drop = tf.compat.v1.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        print(self.h_drop)
        # Final (unnormalized) scores and predictions
        with tf.compat.v1.name_scope("output"):
            # W = tf.get_variable(
            #     "W",
            #     shape=[num_filters_total, num_classes],
            #     initializer=tf.contrib.layers.xavier_initializer())
            W = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            self.WW.append(W)
            #W = tf.get_variable('W_', [num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_classes]), name="b")
            self.bb.append(b)
            #b = tf.get_variable('b_', [num_classes], initializer=tf.contrib.layers.xavier_initializer())
            l2_loss += tf.compat.v1.nn.l2_loss(W)
            l2_loss += tf.compat.v1.nn.l2_loss(b)
            self.scores = tf.compat.v1.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            print(self.scores)
            self.predictions = tf.compat.v1.argmax(self.scores, 1, name="predictions")

        # Calculate cross-entropy loss
        with tf.compat.v1.name_scope("loss"):
            losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.compat.v1.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.compat.v1.name_scope("accuracy"):
            correct_predictions = tf.compat.v1.equal(self.predictions, tf.compat.v1.argmax(self.input_y, 1))
            self.accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_predictions, "float"), name="accuracy")
