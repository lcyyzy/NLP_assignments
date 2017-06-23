#-*- coding: utf8 -*-

import logging
import collections

logging.basicConfig(level=logging.INFO)
loger = logging.getLogger('Load_Message_Data',)

def load_message():
    messages = []
    #recs = []

    with open('data/message.xml') as fi:
        lines = fi.readlines()
        num = len(lines)
        for i in range(0, num, 8):
            content = lines[i+1].strip().strip('"')
            fst_classid = lines[i+5].strip()
            sed_classid = lines[i+6].strip()
            if fst_classid not in ['1','2','3','4']:
                print(content, sed_classid, lines[i])
            messages.append([content, fst_classid, sed_classid])
            #recs.append(lines[i+2].strip().strip('"'))

    fst_counter = collections.Counter(i for _,i,_ in messages)
    sed_counter = collections.Counter(i for _,_,i in messages)

    loger.info("一级分类各类短信数目:"+str(fst_counter))
    loger.info("二级分类各类短信数目:"+str(sed_counter))

    return messages

def load_chaifenzi():
    chaifenzi = {}

    with open('data/chaifenzi.txt', encoding='GBK', errors='ignore') as fi:
        for line in fi:
            try:
                c, r = line.split()
                chaifenzi[c] = r
            except:
                print(line)

    return chaifenzi
chaifenzi = load_chaifenzi()


def load_fantizi():
    fantizi = {}
    with open('data/fantizi.txt', encoding='GBK', errors='ignore') as fi:
        for line in fi:
            try:
                j,f = line.split()
                    #if (f,j) in [('堆','栈'),('太','酞'),('霉','酶')]:
                    #continue
                fantizi[f] = j
                #print(j,f)
            except:
                print(line)
    return fantizi
fantizi = load_fantizi()

def check_chaifen_fanti():
    chaifenzi = load_chaifenzi()
    fantizi = load_fantizi()
    chaifen, fanti = 0, 0
    for i  in content:
        for j in chaifenzi:
            if j in content:
                chaifen += 1
        for k in fantizi:
            if k in content:
                fantizi += 1
    return chaifen, fanti



if __name__ == '__main__':
    messages = load_message()
