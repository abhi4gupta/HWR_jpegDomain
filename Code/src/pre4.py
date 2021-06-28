import random

import cv2
import numpy as np
import numpy
import matplotlib.pyplot as plt
import os
from SamplePreprocessor import preprocess
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc
import matplotlib.pylab as pylab
from matplotlib import pyplot

pylab.rcParams['figure.figsize'] = (20.0, 7.0)

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )


def size(img):
    return tuple(img.shape[1::-1])

def main():
    i = 0
    f = open('../data/gt/words.txt')
    bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
    for line in f:
        if not line or line[0] == '#':
            continue
        lineSplit = line.strip().split(' ')
        assert len(lineSplit) >= 9
        fileNameSplit = lineSplit[0].split('-')
        if not os.path.exists('../data4/img/'+ fileNameSplit[0]):
            os.makedirs('../data4/img/'+ fileNameSplit[0])
        if not os.path.exists('../data4/img/'+ fileNameSplit[0]+'/' + fileNameSplit[0] + '-' + fileNameSplit[1]):
            os.makedirs('../data4/img/'+ fileNameSplit[0]+'/' + fileNameSplit[0] + '-' + fileNameSplit[1])

        fileName = '../data/img/' + fileNameSplit[0]  +'/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'
        img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

        img2 = preprocess(img, (128, 32), False)
        img2 = img2*255


        #dct
        imsize = img2.shape
        dct = np.zeros(imsize)
        for k in r_[:imsize[0]:4]:
            for j in r_[:imsize[1]:4]:
                dct[k:(k+4),j:(j+4)] = dct2( img2[k:(k+4),j:(j+4)] )


        save_path = '../data4/img/' + fileNameSplit[0]  +'/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' +lineSplit[0]+'.png'
        save_path2 = '../data4/img/' + fileNameSplit[0]  +'/' + fileNameSplit[0] + '-' + fileNameSplit[1]
        print(save_path2)
        cv2.imwrite(save_path,dct)

        # cv2.imshow('image',img2)
        # cv2.waitKey(0)

        
        # i = i+1
        # if i==5:
        #     break

if __name__ == '__main__':
    main()



