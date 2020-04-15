# -*- coding: UTF-8 -*-

import pickle,pprint
from PIL import Image
import numpy as np
import os
import matplotlib.image as plimg

class DictSave(object):
    def __init__(self, fileNames, file):
        self.fileNames = fileNames
        self.file      = file
        self.arr       = []
        self.allArr    = []
        self.lable     = []

    def imageInput(self, filenames, file):
        i = 0
        for filename in filenames:
            self.arr, self.lable = self.readFile(filename, file)
            if self.allArr == []:
                self.allArr = self.arr
            else:
                self.allArr = np.concatenate((self.allArr, self.arr))
            print(i)
            i = i + 1

    def readFile(self, filename, file):
        im = Image.open(filename)

        r, g, b = im.split()

        r_arr = plimg.pil_to_array(r)
        g_arr = plimg.pil_to_array(g)
        b_arr = plimg.pil_to_array(b)

        arr = np.concatenate((r_arr, g_arr, b_arr))
        label = []
        for i in file:
            label.append(i[0])
        return arr,label

    def pickleSave(self, arr, label):
        print("正在存储")

        contact = {'data':arr, 'lable':label}

        f = open('data_batch', 'wb')
        pickle.dump(contact, f)
        f.close()

        print("存储完毕")

if __name__ == "__main__":
    file_dir = 'E:\python\\train_data'
    L = []
    F = []
    for root,dirs,files in os.walk(file_dir):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))

        for file in files:
            if os.path.splitext(file)[1] == '.JPG':
                L.append(os.path.join(root, file))
                F.append(file)

    ds = DictSave(L, F)
    ds.imageInput(ds.fileNames, ds.file)
    print(ds.allArr)
    ds.pickleSave(ds.allArr, ds.lable)
    print("最终数组的大小："+str(ds.allArr.shape))
