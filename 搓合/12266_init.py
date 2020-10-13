# -*- coding: utf-8 -*-
T = int(input())

for t in range(T):
    N = int(input())
    buy = []
    sell = []
    stockprice = -1
    for n in range(N):
        str = input()
        strlist = str.split()
        share = int(strlist[1])
        price = int(strlist[-1])

        #TODO:處理交易 

        # 處理列印
        if not sell: # 如果空了
            print('-', end=' ')
        else:
            print(sell[0][0], end=' ')
        if not buy:
            print('-', end=' ')
        else:
            print(buy[0][0], end=' ')
        if stockprice == -1:
            print('-')
        else:
            print(stockprice)