import torch
import pandas as pd
import numpy as np
from importlib import import_module
from sklearn.preprocessing import LabelEncoder
from os import path


le = LabelEncoder()

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

class RequestHandler:

    def __init__(self, model_name):
        cur_path = path.dirname(__file__)
        self.dataset = path.join(cur_path, 'HITSZQA')
        x = import_module('nlp_module.models.' + model_name)
        config = x.Config(self.dataset)
        config.batch_size = 1
        self.device = config.device
        self.tokenizer = config.tokenizer
        self.pad_size = config.pad_size
        self.model = x.Model(config).to(self.device)
        self.model.load_state_dict(torch.load(config.save_path, map_location=self.device.type))
        self.model.eval()
        dataset = pd.read_csv(config.train_path, encoding='utf-8', names=['comments', 'label'], sep='\t', header=None)
        label = np.array(dataset['label'])
        le.fit_transform(label)

    def get_result(self, sentence):
        def inverse_label(predict):
            label = le.inverse_transform([predict])
            return list(label)[0]

        token = [CLS] + self.tokenizer.tokenize(sentence)
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        seq_len = len(token)
        if seq_len < self.pad_size:
            mask = [1] * len(token_ids) + [0] * (self.pad_size - seq_len)
            token_ids += ([0] * (self.pad_size - seq_len))
        else:
            mask = [1] * self.pad_size
            token_ids = token_ids[:self.pad_size]
            seq_len = self.pad_size

        token_ids = torch.LongTensor([token_ids]).to(self.device)
        seq_len = torch.LongTensor([seq_len]).to(self.device)
        mask = torch.LongTensor([mask]).to(self.device)
        text = (token_ids, seq_len, mask)
        output = self.model(text)
        confidence = torch.nn.functional.softmax(output.data, dim=1)
        label_ids = torch.max(confidence, 1)[1].cpu().numpy()
        return inverse_label(label_ids[0])[9:], confidence[0][label_ids[0]].item()


if __name__ == '__main__':
    rh_sub = RequestHandler('bert')
    result = rh_sub.get_result(u'哈工大食堂怎么样？')
    print(result)