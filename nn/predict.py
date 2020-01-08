# -*- coding:utf-8 -*-
# __author__ = 'chy'

import csv

import nn.RequestHandler as RequestHandler

if __name__ == '__main__':
    f = open('test_public.csv', 'r', encoding='UTF-8')
    fw = open('result.csv', 'w')
    lines = csv.reader(f)
    writer = csv.writer(fw)
    next(lines, None)
    # results_list = []
    sen_list = []

    rh_sub = RequestHandler.RequestHandler(1)
    for line in lines:
        if len(line) != 2:  # [id,content,sub.value,word]
            print(line)
            continue
        sub_id = str(line[0])
        sen = line[1].strip()
        sub = rh_sub.getResult(str(sen))
        sen_list.append([sub_id, sen, sub])

    for res in sen_list:
        try:
            writer.writerow(res)
        except:
            print(res)

    # rh_value = RequestHandler.RequestHandler(2)
    # print('###############################')
    # for sentence in sen_list:
    #     value = rh_value.getResult(str(sentence[1]))
    #     tmp_sentence = [sentence[0], sentence[1], sentence[2], value]
    #     results_list.append(tmp_sentence)
    #
    # print(results_list[0])
    # for res in results_list:
    #     try:
    #         writer.writerow(res)
    #     except:
    #         print(res)
    f.close()
    fw.close()
