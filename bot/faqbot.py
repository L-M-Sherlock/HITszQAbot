from os import path

import nonebot

import bot.config as config

if __name__ == '__main__':
    nonebot.init(config)
    nonebot.load_plugins(
        path.join(path.dirname(__file__), 'faq', 'plugins'),
        'faq.plugins',

    )
    nonebot.run()
