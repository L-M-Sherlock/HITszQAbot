import datetime
import os
import pickle as pickle
import time

import tensorflow as tf
import nn.utils as utils
from nn.classifier_cnn import CNNClassifier
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# import sys
# reload(sys)  
# sys.setdefaultencoding('utf-8')
# Parameters
# ==================================================

# # Data loading params
# tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# # Model Hyperparameters
# tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,5,8')")
# tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_reg_lambda", 1.0, "L2 regularization lambda (default: 1.0)loss_multilabel")

# # Training parameters
# tf.flags.DEFINE_integer("cnn_batch_size", 64, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# # Misc Parameters
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

class TrainFlags():
    def __init__(self):
        self.dev_sample_percentage = 0.1
        self.embedding_dim = 128
        self.filter_sizes = "3,4,5"
        self.num_filters = 128
        self.dropout_keep_prob = 0.5
        self.l2_reg_lambda = 1.0
        self.cnn_batch_size = 64
        self.num_epochs = 250
        self.checkpoint_every = 100
        self.evaluate_every = 100
        self.num_checkpoints = 5
        self.allow_soft_placement = True
        self.log_device_placement = False


def Train(floder_path, out_dir=None):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_train, y_train, vocabulary, vocabulary_inv = utils.load_data(floder_path)

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_train))
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.3, random_state=2018)

    # x_train, x_dev = x_train[:dev_sample_index], x_train[dev_sample_index:]
    # y_train, y_dev = y_train[:dev_sample_index], y_train[dev_sample_index:]
    # print("Vocabulary Size: {:d}".format(len(vocabulary)))
    # print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # x_train, y_train, x_dev, y_dev = utils.load_data_word2vec()
    # FLAGS.embedding_dim = 400
    # x_dev = x_train
    # y_dev = y_train
    with open(os.path.join(floder_path, "vocab.pkl"), "wb") as f:
        pickle.dump([vocabulary, vocabulary_inv], f)
    print(x_train.shape, y_train.shape, x_dev.shape, y_dev.shape)
    # rint x_train[0]
    # x_train, y_train = np.vstack((x_train,x_dev)), np.vstack((y_train,y_dev))
    # Training
    # ==================================================
    FLAGS = None
    FLAGS = TrainFlags()
    with tf.compat.v1.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            cnn = CNNClassifier(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocabulary),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.compat.v1.Variable(0, name="global_step", trainable=False)
            optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.compat.v1.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.compat.v1.summary.scalar("{}/grad/sparsity".format(v.name),
                                                                   tf.compat.v1.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.compat.v1.summary.merge(grad_summaries)

            # Output directory for models and summaries
            if not out_dir:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.compat.v1.summary.scalar("loss", cnn.loss)
            acc_summary = tf.compat.v1.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.compat.v1.summary.FileWriter(dev_summary_dir, sess.graph)
            '''
            # Visualization for embedding
            # Write meta
            with codecs.open(os.path.join(out_dir, "metadata.tsv"), 'w', encoding='utf-8') as tsv_file:
                for vocab in vocabulary_inv:
                    tsv_file.write(vocab + "\n")

            config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = cnn.W.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = os.path.join(out_dir, 'metadata.tsv')
            tf.contrib.tensorboard.plugins.projector.visualize_embeddings(tf.summary.FileWriter(dev_summary_dir), config)
            '''
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # TODO: fix this
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn.training: 1
                }
                _, step, summaries, loss, accuracy, W, W0, W1, W2, W3, b0, b1, b2, b3 = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.W, cnn.WW[0], cnn.WW[1],
                     cnn.WW[2], cnn.WW[3], cnn.bb[0], cnn.bb[1], cnn.bb[2], cnn.bb[3]],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.training: 0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy

            batches = utils.batch_iter(
                list(zip(x_train, y_train)), FLAGS.cnn_batch_size, FLAGS.num_epochs)
            score = 0.0
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.compat.v1.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    score = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)

                    # Save the model for Embedding Visualization
                    saver.save(sess, os.path.join(dev_summary_dir, "model.ckpt"), global_step=current_step)

                    print("Saved model checkpoint to {}\n".format(path))
        return checkpoint_dir


if __name__ == "__main__":
    print(Train("./data", "./model"))
