#!/usr/local/bin/python3
import urllib.request
import sys

folder_name = sys.argv[1]
file_name = folder_name + '.txt'
f = open(file_name, 'r')


counter = 1
for line in f:
    if 'jpg' not in line:
        continue
    if counter > 100:
        break
    file_name = folder_name + '/' + str(counter) + '.jpg'
    try:
        urllib.request.urlretrieve(str(line),file_name)
        counter += 1
    except Exception as e:
        #do nothing
        continue
        # print('failure on: '+ line)
