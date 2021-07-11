# encoding:utf-8

from nonebot import on_command
from nonebot.rule import to_me
from nonebot.permission import Permission
from nonebot.typing import T_State
from nonebot.adapters import Bot, Event
from tools.search import search
import nonebot.adapters.cqhttp.message as message

from ..txt_tools import add_at

faq = on_command("", rule=to_me(), permission=Permission(), priority=1)


@faq.args_parser
async def parse(bot: Bot, event: Event, state: T_State):
    print(state["_current_key"], ":", str(event.get_message()))
    state[state["_current_key"]] = str(event.get_message())


@faq.handle()
async def query(bot: Bot, event: Event):
    raw_question = str(event.get_message())
    question = raw_question.replace(' ', '')
    question = question.replace('\r\n', '')
    if question:
        reply = await search_results(question)
        _, group_id, user_id = event.get_session_id().split("_")
        if event.is_tome() and reply and (user_id in bot.config.superusers or group_id in bot.config.groups):
            reply = add_at(reply, event.get_user_id())
            await faq.send(message.Message(reply))


async def search_results(content):
    results = search(content, 3)
    top3 = results[:min(len(results), 3)]
    if len(top3) == 0:
        return None
    text = "\n".join([f"{res.title}：{res.url}" for res in top3])
    return text
