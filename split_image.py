import numpy as np
from matplotlib import pyplot as plt
import os
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
'''
im = cv2.imread('data_space/1.jpg')

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(im, contours, -1, (0,255,0), 3)
'''



'''
def split_im(im_path):
    im = cv2.imread(im_path)

    im_bw = np.zeros((200, 400))

    for i in range(0, 200):
        for j in range(0, 400):
            if np.sum(im[i, j, :]) < 600:
                im_bw[i, j] = 200



#first
    boxm = 0
    xarg = 0
    for x in range(60, 100):
        box = im_bw[50:150, x:(x + 50)]
        if np.mean(box) > boxm:
            xarg = x
            boxm = np.mean(box)
       

    first_image  = im[50:150, xarg:(xarg+50)]

    boxm = 0
    xarg = 0
    for x in range(xarg + 80, 160):
        box = im_bw[50:150, x:(x + 50)]
        if np.mean(box) > boxm:
            xarg = x
            boxm = np.mean(box)
        
    second_image = im[50:150, xarg:(xarg+50)]

#third


    boxm = 0
    xarg = 0
    for x in range(xarg+80, 210):
        box = im_bw[50:150, x:(x + 50)]
        if np.mean(box) > boxm:
            xarg = x
            boxm = np.mean(box)
       
    third_image = im[50:150, xarg:(xarg+50)]



    boxm = 0
    xarg = 0
    for x in range(xarg + 80, 300):
        box = im_bw[50:150, x:(x + 50)]
        if np.mean(box) > boxm:
            xarg = x
            boxm = np.mean(box)
       
    fourth_image = im[50:150, xarg:(xarg+50)]
    return first_image, second_image, third_image, fourth_image
'''




def split_im(data_path):
    im = cv2.imread(data_path)


    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 200, 255, 0)
    c_inds = []
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contlen = []
    for i in range(0, len(contours)):
        #if len(contours[i]) > 45:
            #c_inds.append(i)
            #print(i)
        contlen.append(len(contours[i]))
    c_inds = np.argsort(contlen)[-4:]
    images = []
    
    minl_list = []
    min_inds = np.arange(0, 4, 1)
    for ind in c_inds:
        cont = contours[ind]
        minl = np.min(contours[ind][:, 0][:, 1])
        minr = np.max(contours[ind][:, 0][:, 1])
        mind = np.min(contours[ind][:, 0][:, 0])
        minu = np.max(contours[ind][:, 0][:, 0])
        minl_list.append(mind)
        resid = 65 - (minr - minl)
        if resid > 0:
            minr = minr + resid
        resid = 35 - (minu - mind)
        if resid > 0 :
            minu = minu + resid
        print(minl -minr, mind- minu)
        images.append(im[minl:minr, mind:minu, :])
        
    print(minl_list) 
    print(min_inds)
    pairs = zip(minl_list, min_inds)
    im_inds = [x[1] for x in sorted(pairs)]
    images = [images[i] for i in im_inds]
    return images

labfi = []
s_dir = 'split_data/'
j = 0
im_c = 0
#labels = np.loadtxt('data/labels.txt')


lab_fil = 'data_space/labels.txt'

fi = open(lab_fil, 'r')

labels = []
for li in fi:
    lab = li.strip()
    labels.append(lab)



for fi in os.listdir('data_space'):
    if fi[-3:] == 'jpg':
        print(fi)
        sys.stdout.flush()
        images  = split_im('data_space/' + fi)
        im1 = images[0]
        im2 = images[1]
        im3 = images[2]
        im4 = images[3]
        plt.imsave( s_dir + str(im_c) + '.jpg'  , im1)
        im_c += 1

        plt.imsave( s_dir + str(im_c) + '.jpg'  , im2)
        im_c +=1 
        plt.imsave( s_dir + str(im_c) + '.jpg'  , im3)
        im_c +=1        
        plt.imsave( s_dir + str(im_c) + '.jpg'  , im4)
        im_c +=1
        labfi.append(labels[j][0])
        labfi.append(labels[j][1])
        labfi.append(labels[j][2])
        labfi.append(labels[j][3])
        j += 1





with open(s_dir + 'labels.txt','w') as f:
    for lb in labfi:
        f.write('%s\n' % lb)
