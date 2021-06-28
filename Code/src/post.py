import cv2
import numpy as np
from SamplePreprocessor import preprocess
import scipy
import scipy.fftpack

from Model import Model

from numpy import zeros
from numpy import r_
import sys
import matplotlib.pylab as pylab

pylab.rcParams['figure.figsize'] = (20.0, 7.0)

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def main(argv):
    print("usage: post.py <inputfilename>")
    i = 0
    fileName = '../dump/'+argv
    # img2 = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    img2 = preprocess(cv2.imread(fileName, cv2.IMREAD_GRAYSCALE), Model.imgSize)

    # img2 = preprocess(img, (128, 32), False)
    img2 = img2*255

    # cv2.imwrite('../dump/out.png'+argv,img2)

    #dct
    imsize = img2.shape
    dct = np.zeros(imsize)
    for k in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct[k:(k+8),j:(j+8)] = dct2( img2[k:(k+8),j:(j+8)] )

    save_path = '../dump2/dct_' + argv
    print('DCT converted image saved at ',save_path)
    cv2.imwrite(save_path,dct)
    save_path = '../dump2/' + 'test.png'
    print('DCT converted image saved at ',save_path)
    cv2.imwrite(save_path,dct)

if __name__ == '__main__':
	main(sys.argv[1])



