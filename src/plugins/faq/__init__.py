# encoding:utf-8
from os import path

from nonebot import on_command
from nonebot.rule import to_me
from nonebot.permission import Permission
from nonebot.typing import T_State
from nonebot.adapters import Bot, Event
import nonebot.adapters.cqhttp.message as message

import config as config
from nlp_module.RequestHandler import RequestHandler
from ..txt_tools import raw_to_answer, add_at

faq = on_command("问答", rule=to_me(), permission=Permission(), priority=5)

Q2A_dict = {}
log_list = []
ans_path = path.join(path.dirname(__file__), 'answers.txt')

with open(ans_path, 'r', encoding='UTF-8-sig') as file:
    lines = file.readlines()
    for line in lines:
        tmp_list = line.split('\t')
        Q2A_dict[tmp_list[0]] = tmp_list[1]
rh_sub = RequestHandler('bert')


@faq.args_parser
async def parse(bot: Bot, event: Event, state: T_State):
    print(state["_current_key"], ":", str(event.get_message()))
    state[state["_current_key"]] = str(event.get_message())


@faq.handle()
async def faq_local(bot: Bot, event: Event):
    raw_question = str(event.get_message())
    question = raw_question.replace(' ', '')
    question = question.replace('\r\n', '')
    if question:
        reply, confidence = await test_local(question, event.is_tome())
        if event.is_tome():
            if confidence < config.CONFIDENCE:
                reply = '我现在还不太明白，但没关系，以后的我会变得更强呢！'
            reply = add_at(reply, event.get_user_id())
            await faq.send(message.Message(reply))


async def test_local(content, callme):
    ans, confidence = rh_sub.get_result(content)
    log = content.replace(',', '，') + ',__label__' + ans + ',' + str(round(confidence, 2)) + ',' + str(
        int(callme)) + '\n'  # 记录问题和预测标签、置信度
    global log_list
    log_list.append(log.encode('GBK'))  # 保存日志到 log_list
    if len(log_list) >= config.LOG_SAVE_LEN:
        log_save()  # 日志长度大等于 LOG_SAVE_LEN 时，写入文件
    ans = Q2A_dict[ans]
    ans = raw_to_answer(ans)
    return ans, confidence


def log_save():
    global log_list
    log_path = path.join(path.dirname(__file__), 'log.csv')
    f = open(log_path, 'ab+')
    f.writelines(log_list)
    log_list = []
    f.close()
