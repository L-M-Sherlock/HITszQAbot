import pytz
import random
from .paratranz_api import get
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

def get_files_list():
    file_list = get("files").json()
    count = 0
    for row in tqdm(file_list):
        if count == 0:
            lsr_csv = pd.DataFrame(columns=row.keys())
            lsr_csv = lsr_csv.append(row, ignore_index=True)
            count += 1
        else:
            lsr_csv = lsr_csv.append(row, ignore_index=True)
    lsr_csv.to_csv(f'./file_list.csv', index=False)

def calc_priority():
    file_list = pd.read_csv("./file_list.csv", index_col=None)
    file_list["开工日期"] = file_list["createdAt"].map(
        lambda x: pd.to_datetime(x).tz_convert('UTC'))
    today = datetime.now(pytz.UTC)
    file_list["等待天数"] = today - file_list["开工日期"]
    file_list["等待天数"] = file_list["等待天数"].map(lambda x: x.days)

    file_list["need_translated_ratio"] = 1 - file_list["translated"] / file_list["total"]
    file_list["need_checked_ratio"] = 1 - file_list["checked"] / file_list["total"]
    file_list["need_reviewed_ratio"] = 1 - file_list["reviewed"] / file_list["total"]

    file_list["工作量"] = round(file_list["words"] * (file_list["need_translated_ratio"]*1+file_list["need_checked_ratio"]*0.5+file_list["need_reviewed_ratio"]*0.2))
    file_list["优先级"] = round((file_list["等待天数"] + file_list["工作量"]) / file_list["工作量"], 4)
    need_work = file_list[(file_list["优先级"] < np.inf) & (file_list["locked"] == 0)].sort_values("优先级", ascending=False)
    need_work["翻译链接"] = need_work["id"].map(lambda x: f"https://paratranz.cn/projects/3131/strings?file={x}")
    need_work["开工日期"] = need_work["开工日期"].map(lambda x: x.strftime("%Y/%m/%d"))
    need_work.drop(columns=["createdAt", "updatedAt", "modifiedAt", "project", "format", "total", "translated", "disputed", "checked", "reviewed", "hidden", "locked", "words", "hash", "extra", "folder", "progress", "need_translated_ratio", "need_checked_ratio", "need_reviewed_ratio"], inplace=True)
    need_work.to_csv(f'./task_list.csv', index=False)

def random_task():
    try:
        task_list = pd.read_csv("./task_list.csv", index_col=None)
    except FileNotFoundError:
        print("FileNotFound")
        get_files_list()
        calc_priority()
        task_list = pd.read_csv("./task_list.csv", index_col=None)
    
    return task_list.loc[weight_choice(task_list['优先级']),'id']

def get_result(file_id):
    file_info = get(f'files/{file_id}').json()

    name = file_info['name']
    total = file_info['total']
    translated = file_info['translated']
    checked = file_info['checked']
    reviewed = file_info['reviewed']
    result = f'{name}\nhttps://paratranz.cn/projects/3131/strings?file={file_id}\n总共{total}条，还剩{total-translated}条待翻译，{translated-checked}条待检查，{checked-reviewed}条待审核～'
    return result

def weight_choice(weight):
    """
    :param weight: list对应的权重序列
    :return:选取的值在原列表里的索引
    """
    t = random.randint(0, round(sum(weight) - 1))
    for i, val in enumerate(weight):
        t -= val
        if t < 0:
            return i    

if __name__ == "__main__":
    get_files_list()
    calc_priority()
    print(get_result(random_task()))
    print(1)
