# -*- coding: utf-8 -*-
# import sys
import os
import tensorflow as tf
import nn.utils as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# reload(sys)
# sys.setdefaultencoding('utf-8')


class RequestHandler():

    def __init__(self, number):
        # conf = configparser.ConfigParser()
        # conf.read("./conf.ini")
        # modular_name = conf.get("reClassify","modular_name")
        # modular_num = conf.get("reClassify","modular_num")
        # tmp_data_path = conf.get("reClassify","data_path")
        # self.data_path = modular_name + modular_num + tmp_data_path
        # self.checkpoint_dir = checkpoint_dir
        # self.checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
        if number == 1 or number == '1':
            self.checkpoint_dir = os.path.abspath(os.path.join("./", "model/checkpoints"))
            self.data_dir = os.path.abspath(os.path.join("./", "data"))
        else:
            self.checkpoint_dir = os.path.abspath(os.path.join("./", "model_value/checkpoints"))
            self.data_dir = os.path.abspath(os.path.join("./", "data_value"))
        self.checkpoint_file = tf.compat.v1.train.latest_checkpoint(self.checkpoint_dir)
        self.graph = tf.compat.v1.Graph()

        self.graph = tf.compat.v1.Graph()
        # print self.data_dir
        self.x, self.y, self.vocabulary, self.vocabulary_inv = utils.load_data(self.data_dir)
        with self.graph.as_default():
            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            sess_config.gpu_options.per_process_gpu_memory_fraction = 0.3

            self.sess = tf.compat.v1.Session(config=sess_config)
            with self.sess.as_default():
                self.saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                self.saver.restore(self.sess, self.checkpoint_file)
                self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
                self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                self.predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]

    def getResult(self, sentence):
        """1. Complete the classification in this function.

        Args:
            sentence: A string of sentence.

        Returns:
            classification: A string of the result of classification.
        """
        raw_x = utils.sentence_to_index(sentence, self.vocabulary, self.x.shape[1])
        predicted_results = self.sess.run(self.predictions, {self.input_x: raw_x, self.dropout_keep_prob: 1.0})
        # rint sentence
        # print '----------------------------------'
        # print raw_x
        # print self.predictions
        # print predicted_results
        return utils.inverse_label(predicted_results[0])[9:]
        # return utils.inverse_label(predicted_results[0])
        # return "chat"

    def getBatchResults(self, sentencesList):
        """2. You can also complete the classification in this function,
                if you want to classify the sentences in batch.

        Args:
            sentencesList: A List of Dictionaries of ids and sentences,
                like:
                [{'id':331, 'content':'帮我打电话给张三' }, 
                 {'id':332, 'content':'帮我订一张机票!' },
                 ... ]

        Returns:
            resultsList: A List of Dictionaries of ids and results.
                The order of the list must be the same as the input list,
                like:
                [{'id':331, 'result':'telephone' }, 
                 {'id':332, 'result':'flight' },
                 ... ]
        """
        resultsList = []
        for sentence in sentencesList:
            resultDict = {}
            resultDict['id'] = sentence['id']
            resultDict['result'] = self.getResult(sentence['content'])
            resultsList.append(resultDict)

        return resultsList


if __name__ == '__main__':
    rh_sub = RequestHandler(1)
    print(rh_sub.getResult(u'毕业生的就业率怎么样'))
