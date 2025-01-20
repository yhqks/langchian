"""

 Created on 2025/1/13  
 @author: yks 
 Path: D:/yhq/python31105\bilibili.py  
"""

from bilibili_api import *

v = video.Video("BV1W94y1i7C1")  # 初始化视频对象

sync(ass.make_ass_file_danmakus_protobuf(
    obj=v, # 生成弹幕文件的对象
    page=0, # 哪一个分 P (从 0 开始)
    out="test.ass" # 输出文件地址
))
