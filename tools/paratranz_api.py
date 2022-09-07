import requests
from nonebot import config

ApiKey = config.Env().apikey

headers = {
    'Content-type': 'application/json',
    'Authorization': ApiKey
}

PREFIX = 'https://paratranz.cn/api/projects/3131/'


def get(suffix):
    return requests.get(PREFIX + suffix, headers=headers)
