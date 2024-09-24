# -*- coding: utf-8 -*-
"""lab2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BaSY6dZaNaOFlJJUxUZGr-UsXusZoLZm
"""

import cv2
import sys
import numpy as np
from google.colab.patches import cv2_imshow

filename = '/content/circle.jpg'
circle = cv2.imread(filename)

filename = '/content/line.jpg'
line = cv2.imread(filename)


width_c  = circle.shape[1]
height_c = circle.shape[0]

width_l  = line.shape[1]
height_l = line.shape[0]

#Cabeça
head = cv2.bitwise_not(circle)
scale_circle = np.float32([[1,0,0],[0,1,0]])
head = cv2.warpAffine(head, scale_circle, (width_c*3,height_c*3))
M_translation_c = np.float32([[1,0,100],[0,1,0]])
im_translated_c = cv2.warpAffine(head,M_translation_c,(width_c*3,height_c*3))
head = cv2.bitwise_not(im_translated_c)

body = cv2.bitwise_not(line)
scale_line = np.float32([[1,0,0],[0,1,0]])
body = cv2.warpAffine(body, scale_line, (width_l*3,height_l*3))
M_rotation_l = cv2.getRotationMatrix2D((width_l/2, height_l/2), 90, 1)
body = cv2.warpAffine(body, M_rotation_l, (width_l*3,height_l*3))
M_translation_l = np.float32([[1,0,100],[0,1,62]])
body = cv2.warpAffine(body,M_translation_l,(width_l*3,height_l*3))
body = cv2.bitwise_not(body)

#braço esquerdo
ba1 = cv2.bitwise_not(line)
scale_ba1 = np.float32([[.75,0,0],[0,.75,0]])
ba1 = cv2.warpAffine(ba1, scale_ba1, (width_l*3,height_l*3))
M_rotation_ba1 = cv2.getRotationMatrix2D((width_l/2, height_l/2), 45, 1)
ba1 = cv2.warpAffine(ba1, M_rotation_ba1, (width_l*3,height_l*3))
M_translation_ba1 = np.float32([[1,0,95],[0,1,62]])
ba1 = cv2.warpAffine(ba1,M_translation_ba1,(width_l*3,height_l*3))
ba1 = cv2.bitwise_not(ba1)

#braço direito
ba2 = cv2.bitwise_not(line)
scale_ba2 = np.float32([[.75,0,0],[0,.75,0]])
ba2 = cv2.warpAffine(ba2, scale_ba2, (width_l*3,height_l*3))
M_rotation_ba2 = cv2.getRotationMatrix2D((width_l/2, height_l/2), -45, 1)
ba2 = cv2.warpAffine(ba2, M_rotation_ba2, (width_l*3,height_l*3))
M_translation_ba2 = np.float32([[1,0,120],[0,1,79]])
ba2 = cv2.warpAffine(ba2,M_translation_ba2,(width_l*3,height_l*3))
ba2 = cv2.bitwise_not(ba2)

#perna esquerda
pe1 = cv2.bitwise_not(line)
scale_pe1 = np.float32([[1.5,0,0],[0,1.5,0]])
pe1 = cv2.warpAffine(pe1, scale_pe1, (width_l*3,height_l*3))
M_rotation_pe1 = cv2.getRotationMatrix2D((width_l/2, height_l/2), 45, 1)
pe1 = cv2.warpAffine(pe1, M_rotation_pe1, (width_l*3,height_l*3))
M_translation_pe1 = np.float32([[1,0,22],[0,1,145]])
pe1 = cv2.warpAffine(pe1,M_translation_pe1,(width_l*3,height_l*3))
pe1 = cv2.bitwise_not(pe1)

#perna direita
pe2 = cv2.bitwise_not(line)
scale_pe2 = np.float32([[1.5,0,0],[0,1.5,0]])
pe2 = cv2.warpAffine(pe2, scale_pe2, (width_l*3,height_l*3))
M_rotation_pe2 = cv2.getRotationMatrix2D((width_l/2, height_l/2), -45, 1)
pe2 = cv2.warpAffine(pe2, M_rotation_pe2, (width_l*3,height_l*3))
M_translation_pe2 = np.float32([[1,0,142],[0,1,111]])
pe2 = cv2.warpAffine(pe2,M_translation_pe2,(width_l*3,height_l*3))
pe2 = cv2.bitwise_not(pe2)

im = cv2.bitwise_and(head, body)
im = cv2.bitwise_and(im, ba1)
im = cv2.bitwise_and(im, ba2)
im = cv2.bitwise_and(im, pe1)
im = cv2.bitwise_and(im, pe2)

cv2_imshow(im)

