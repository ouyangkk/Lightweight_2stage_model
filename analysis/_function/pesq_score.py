#_*conding:utf-8_*_
# __author__ = YuyongKang, NenghengZheng
# __date__ = '2020/5/27'
# __filename__ = 'pesq_score.py'
# __IDE__ = 'PyCharm'
# __copyright__ = Shenzhen University ,Electronic and infromation college,Intelligent speech and artificial hearing Lab

import numpy
import librosa
from scipy.io import wavfile
import os
import sys
def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False

def pesq(obj_path, refer_path):
    path = os.getcwd()
    pesq_cmd = os.path.join(path,'_function', 'pesq')

    rate = '+'+str(16000)
    a = os.popen(pesq_cmd + ' ' + rate + ' ' + refer_path + ' ' + obj_path).read()
    lines = a
    pesq = lines[-6:-1]

    if is_number(pesq):
        pesq = float(pesq)
    else:
        pesq = None
    return pesq

