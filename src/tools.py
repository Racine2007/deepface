#coding:utf-8

from crawler import *

def result_2_rect(result, width, height):
    x0 = width*result[0]
    y0 = height*result[1]
    x1 = width*result[2]
    y1 = height*result[3]

    return Rect(int(x0),int(y0),int(x1-x0+1),int(y1-y0+1))
