import os
import cv2
import cfg
from Synthtext.gen_useoriginal import datagen
import time
import requests, datetime, time
import threading
import numpy as np
import os
# from Synthtext.gen_new import datagen, multiprocess_datagen

datagener = datagen()

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    

    i_s_dir = os.path.join(cfg.data_dir, cfg.i_s_dir)
    mask_t_dir = os.path.join(cfg.data_dir, cfg.mask_t_dir)

    
    makedirs(i_s_dir)
    makedirs(mask_t_dir)


    # calculate the len of creating images number to give a name to image   eg: 000001  1-99999 total is 100000
    digit_num = len(str(cfg.sample_num)) - 1

    for idx in range(0,200000):
        print ("Generating step {:>6d} / {:>6d}".format(idx + 1, cfg.sample_num))

        i_s_generate = True
        classRate1 = np.random.rand()
        classRate=[classRate1,classRate1]

        # print("classRate",classRate)
        # i_s, mask_t = datagener.gen_srnet_data_with_background(classRate)

        try:

            i_s,mask_t = datagener.gen_srnet_data_with_background(classRate)

        except:

                    continue

        # i_s = cv2.resize(i_s, (256, 64))
        # mask_t = cv2.resize(mask_t, (256, 64))


        # image resize
        h = i_s.shape[0]
        w = i_s.shape[1]
        ratio = 64 / h * w
        if ratio > 256:
            i_s = cv2.resize(i_s, (256, 64))
            mask_t = cv2.resize(mask_t, (256, 64))

        else:
            # print("fffffffffffffffffffffffffffff")
            ratioInt = int(ratio)
            imageContainer_image = np.zeros((64, 256, 3))
            imageContainer_image_mask = np.zeros((64, 256))
            i_s = cv2.resize(i_s, (ratioInt, 64))
            mask_t = cv2.resize(mask_t, (ratioInt, 64))

            # i_s setMiddle
            point = i_s[0][ratioInt - 1]
            leftLength = (int)((256 - ratioInt) / 2)
            imageContainer_image[:, leftLength:(ratioInt + leftLength), :] = i_s
            imageContainer_image[:, ratioInt + leftLength:] = point
            imageContainer_image[:, :leftLength] = point
            i_s = imageContainer_image

            # mask_t setMiddle
            point = mask_t[0][ratioInt - 1]
            leftLength = (int)((256 - ratioInt) / 2)
            imageContainer_image_mask[:, leftLength:(ratioInt + leftLength)] = mask_t
            imageContainer_image_mask[:, ratioInt + leftLength:] = point
            imageContainer_image_mask[:, :leftLength] = point
            mask_t = imageContainer_image_mask

        i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
        mask_t_path = os.path.join(cfg.data_dir, cfg.mask_t_dir, str(idx).zfill(digit_num) + '.png')

        cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_t_path, mask_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])








if __name__ == '__main__':
    main()
