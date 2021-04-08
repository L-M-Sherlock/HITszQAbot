import re


def cqp_replace(matched):
    f = "[CQ:image,file="
    i = ".png]"
    result = ""
    matched_str = matched.group()
    num = int(re.findall(r'\d+', matched_str)[1])
    for img in range(2, num + 2):
        result = result + f + re.findall(r'\d+', matched_str)[img] + i
    return result + '\\n'


def raw_to_answer(raw):
    rule = "%img\d%\(\d\)(\s\d+)+"
    ans = re.sub(rule, cqp_replace, raw)
    ans = ans.replace("\\n", "\n")
    return ans


def add_at(reply, user_id):
    at = "[CQ:at,qq="
    at += str(user_id)
    at += "]\n"
    at += reply
    return at

