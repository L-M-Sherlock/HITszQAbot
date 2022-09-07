# encoding:utf-8

from nonebot import on_command
from nonebot.rule import to_me
from nonebot.permission import Permission, SUPERUSER
from nonebot.typing import T_State
from nonebot.adapters import Bot, Event
import nonebot.adapters.cqhttp.message as message
from nonebot.log import logger
from tools.create_task import *

newtask = on_command("task", rule=to_me(), aliases={"上工", "推荐上工"}, permission=Permission(), priority=1)


@newtask.args_parser
async def parse(bot: Bot, event: Event, state: T_State):
    logger.info(state["_current_key"], ":", str(event.get_message()))
    state[state["_current_key"]] = str(event.get_message())


@newtask.handle()
async def query(bot: Bot, event: Event):
    reply = await new_task()
    result = event.get_session_id().split("_")
    logger.info(result)
    if len(result) == 1:
        user_id = result[0]
        group_id = ""
    else:
        _, group_id, user_id = result
    if event.is_tome() and reply and (user_id in bot.config.superusers or group_id in bot.config.groups):
        # reply = add_at(reply, event.get_user_id())
        # logger.info(reply)
        await newtask.send(message.Message(reply))

async def new_task():
    return get_result(random_task())


updatetask = on_command("new", rule=to_me(), aliases={"更新任务"}, permission=SUPERUSER, priority=5)

@newtask.handle()
async def update_task(bot: Bot, event: Event):
    get_files_list()
    calc_priority()
    await newtask.send(message.Message("更新成功"))