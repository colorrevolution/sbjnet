import os
import cv2
import cfg
# from Synthtext.gen_standard import datagen
from Synthtext.gen_standard_original_generate_incomplete import datagen

# from Synthtext.gen import datagen, multiprocess_datagen
# from Synthtext.gen_new import datagen, multiprocess_datagen

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # i_t_dir = os.path.join(cfg.data_dir, cfg.i_t_dir)
    i_s_dir = os.path.join(cfg.data_dir, cfg.i_s_dir)
    # t_sk_dir = os.path.join(cfg.data_dir, cfg.t_sk_dir)
    t_t_dir = os.path.join(cfg.data_dir, cfg.t_t_dir)
    t_b_dir = os.path.join(cfg.data_dir, cfg.t_b_dir)
    # t_f_dir = os.path.join(cfg.data_dir, cfg.t_f_dir)
    mask_t_dir = os.path.join(cfg.data_dir, cfg.mask_t_dir)
    full_image_dir = os.path.join(cfg.data_dir, cfg.full_image_dir)
    #
    # makedirs(i_t_dir)
    makedirs(i_s_dir)
    # makedirs(t_sk_dir)
    makedirs(t_t_dir)
    makedirs(t_b_dir)
    # makedirs(t_f_dir)
    makedirs(mask_t_dir)
    makedirs(full_image_dir)

    datagener = datagen()
    fileRecord = open("/media/mmsys9/系统/xff/TextErase/SRNet-master/SRNet-master/SRNet-Datagen/using_dataset_1064fonts_update_hull.txt", "a", encoding="utf-8")


    # calculate the len of creating images number to give a name to image   eg: 000001  1-99999 total is 100000
    # digit_num = len(str(cfg.sample_num)) - 1
    digit_num = len(str(cfg.sample_num)) - 1
    # ratio_number_List = [[0,0,0,100000],[0,25%,100000,180000],[25%,50%,180000,260000],[50,75%,260000,340000],[75%,100%,340000,420000],[100%,100%,420000,500000]]
    ratio_number_List = [[0, 0.25, 101301, 180000], [0.25, 0.5, 180000, 260000],[0.5, 0.75, 260000, 340000], [0.75, 1, 340000, 420000], [1, 1, 420000, 500000]]
    # ratio_number_List = [[0,0,0,100000]]
    for ratio_number in ratio_number_List:
        classRate = [ratio_number[0],ratio_number[1]]
        digit_num = 6
        idx  = ratio_number[2]
        end = ratio_number[3]
        while idx < end:
            print("Generating step {:>6d} / {:>6d}".format(idx, cfg.sample_num))

            # i_t, i_s, t_sk, t_t, t_b, t_f, mask_t,full_image = mp_gen.dequeue_data()
            # i_t, i_s, t_sk, t_t, t_b, t_f, mask_t,text,full_image = datagener.gen_srnet_data_with_background()
            # i_t, i_s, t_sk, t_t, t_b, t_f, surf2 = datagener.gen_srnet_data_with_background()

            try:
                t_t,full_image,i_s,mask_t,t_b,text,font = datagener.gen_srnet_data_with_background(classRate)
            except:
                continue
            # print("dddddddddddddddddddddddddddddd")
            fileRecord.write(str(idx).zfill(digit_num) + '.png'+"    "+str(text)+"    "+str(font)+"\n")
            # i_t_path = os.path.join(i_t_dir, str(idx).zfill(digit_num) + '.png')
            i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
            # t_sk_path = os.path.join(t_sk_dir,"1"+ str(idx).zfill(digit_num) + '.png')
            t_t_path = os.path.join(t_t_dir, str(idx).zfill(digit_num) + '.png')
            t_b_path = os.path.join(t_b_dir, str(idx).zfill(digit_num) + '.png')
            # t_f_path = os.path.join(t_f_dir, str(idx).zfill(digit_num) + '.png')
            mask_t_path = os.path.join(cfg.data_dir, cfg.mask_t_dir, str(idx).zfill(digit_num) + '.png')
            # test_path = os.path.join(cfg.data_dir, cfg.test, str(idx).zfill(digit_num) + '.png')
            full_image_path = os.path.join(full_image_dir, str(idx).zfill(digit_num) + '.png')
            # # print("i_t_path",i_t[0].shape)
            # cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # cv2.imwrite(t_sk_path, t_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            cv2.imwrite(t_t_path, t_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            cv2.imwrite(t_b_path, t_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # cv2.imwrite(t_f_path, t_f, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            cv2.imwrite(mask_t_path, mask_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            cv2.imwrite(full_image_path, full_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # cv2.imwrite(test_path, mask_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            idx = idx + 1



if __name__ == '__main__':
    main()
