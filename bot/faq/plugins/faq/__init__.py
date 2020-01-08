# encoding:utf-8
import json
import bot.config as config
from typing import Optional
import aiohttp
from aiocqhttp.message import escape
from nonebot import on_command, CommandSession
from nonebot import on_natural_language, NLPSession, IntentCommand
# import requests
import string
import re
from nonebot.helpers import context_id, render_expression
import nn.RequestHandler as RH

EXPR_DONT_UNDERSTAND = (
    '我现在还不太明白，但没关系，以后的我会变得更强呢！',
    '我有点不懂你的意思呀，可以问我其他招生相关的问题嘛',
    '这个问题我还不懂哦，问问老师吧~',
    '抱歉哦，我目前还没有这个问题的答案，但我会加油的～'
)


@on_command('faq')
async def faq_local(session: CommandSession):
    question = session.state.get('message')
    replay = await test_local(question)
    await session.send(replay)


# async def faq(session: CommandSession):
#     question = session.state.get('message')
#     at = "[CQ:at,qq="
#     at += str(session.ctx.get('user_id'))
#     at += "]\n"
#     reply = await test(question)
#     if reply:
#         if reply == {}:
#             reply = {'1': ['', '']}
#         confidence = float(list(reply.keys())[0])
#         answer = list(reply.values())[0][1]
#         if confidence < config.CONFIDENCE:
#             rule = "%img\d%\(\d\)(\s\d+)+"
#             answer = re.sub(rule, cqp_replace, answer)
#             answer = answer.replace("\\n", "\n")
#             at += answer
#         else:
#             reply = await call_tuling_api(session, question)
#             if "图灵" in reply or "限制" in reply:
#                 random = render_expression(EXPR_DONT_UNDERSTAND)
#                 at += random
#             elif reply:
#                 at += reply
#             else:
#                 random = render_expression(EXPR_DONT_UNDERSTAND)
#                 at += random
#     else:
#         reply = await call_tuling_api(session, question)
#         if "图灵" in reply or "限制" in reply:
#             random = render_expression(EXPR_DONT_UNDERSTAND)
#             at += random
#         else:
#             at += reply
#     await session.send(at)

'''
    else:
        random = render_expression(EXPR_DONT_UNDERSTAND)
        at += random
        await session.send(at)
'''


@on_natural_language
async def _(session: NLPSession):
    if str(session.ctx.get('sub_type')) == 'normal' and str(session.ctx.get('group_id')) in config.while_list:
        return IntentCommand(80.0, 'faq', args={'message': session.msg_text})


# async def test(message):
#     url = config.API
#     dic = dict()
#     dic['question'] = message
#     headers = {'Content-type': 'application/json'}
#     try:
#         r = requests.post(url, data=json.dumps(
#             dic), headers=headers, timeout=15000)
#         if r.status_code == 200:
#             data = r.json()
#             # results_list = data['k']
#             return data
#         else:
#             print("wrong,status_code: ", r.status_code)
#             return None
#     except Exception as e:
#         print(Exception, ' : ', e)
#         return None

async def test_local(message):
    return RH.RequestHandler(1).getResult(message)


def cqp_replace(matched):
    f = "[CQ:image,file="
    i = ".png]"
    result = ""
    matched_str = matched.group()
    num = int(re.findall(r'\d+', matched_str)[1])
    for img in range(2, num + 2):
        result = result + f + re.findall(r'\d+', matched_str)[img] + i
    return result + '\\n'


async def call_tuling_api(session: CommandSession, text: str) -> Optional[str]:
    # 调用图灵机器人的 API 获取回复

    if not text:
        return None

    url = 'http://openapi.tuling123.com/openapi/api/v2'

    # 构造请求数据
    payload = {
        'reqType': 0,
        'perception': {
            'inputText': {
                'text': text
            }
        },
        'userInfo': {
            'apiKey': session.bot.config.TULING_API_KEY,
            'userId': context_id(session.ctx, use_hash=True)
        }
    }

    group_unique_id = context_id(session.ctx, mode='group', use_hash=True)
    if group_unique_id:
        payload['userInfo']['groupId'] = group_unique_id

    try:
        # 使用 aiohttp 库发送最终的请求
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=payload) as response:
                if response.status != 200:
                    # 如果 HTTP 响应状态码不是 200，说明调用失败
                    return None

                resp_payload = json.loads(await response.text())
                if resp_payload['results']:
                    for result in resp_payload['results']:
                        if result['resultType'] == 'text':
                            # 返回文本类型的回复
                            return result['values']['text']
    except (aiohttp.ClientError, json.JSONDecodeError, KeyError):
        # 抛出上面任何异常，说明调用失败
        return None
