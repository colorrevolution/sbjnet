import os
import cv2
import cfg
from Synthtext.gen import datagen
import time
import requests, datetime, time
import threading
import numpy as np
import os
# from Synthtext.gen_new import datagen, multiprocess_datagen

datagener = datagen()

# over time detect
class MyThread(threading.Thread):
    def __init__(self, target, args=()):
        """
        why: 因为threading类没有返回值,因此在此处重新定义MyThread类,使线程拥有返回值
        此方法来源 https://www.cnblogs.com/hujq1029/p/7219163.html?utm_source=itdadao&utm_medium=referral
        """
        super(MyThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        # 接受返回值
        self.result = self.func(*self.args)

    def get_result(self):
        # 线程不结束,返回值为None
        try:
            return self.result
        except Exception:
            return None


# 为了限制真实请求时间或函数执行时间的装饰器
def limit_decor(limit_time):
    """
    :param limit_time: 设置最大允许执行时长,单位:秒
    :return: 未超时返回被装饰函数返回值,超时则返回 None
    """

    def functions(func):
        # 执行操作
        def run(*params):
            thre_func = MyThread(target=func, args=params)
            # 主线程结束(超出时长),则线程方法结束
            thre_func.setDaemon(True)
            thre_func.start()
            # 计算分段沉睡次数
            sleep_num = int(limit_time // 1)
            sleep_nums = round(limit_time % 1, 1)
            # 多次短暂沉睡并尝试获取返回值
            for i in range(sleep_num):
                time.sleep(1)
                infor = thre_func.get_result()
                if infor:
                    return infor
            time.sleep(sleep_nums)
            # 最终返回值(不论线程是否已结束)
            if thre_func.get_result():
                return thre_func.get_result()
            else:
                return "请求超时"  #超时返回  可以自定义

        return run

    return functions

#接口函数
def a1():
    print("开始请求接口")

    #这里把逻辑封装成一个函数,使用线程调用
    a_theadiing = MyThread(target=a2)
    a_theadiing.start()
    a_theadiing.join()

    #返回结果
    a = a_theadiing.get_result()

    print("请求完成")
    return a
@limit_decor(10)   #超时设置为10s   2s逻辑未执行完毕返回接口超时
def a2():
    try :
        i_s, t_t, mask_t, text = datagener.gen_srnet_data_with_background()
    except:
        a = "chuxiancuowu"
        return a
    a = [i_s, t_t, mask_t, text]
    return  a


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    
    i_t_dir = os.path.join(cfg.data_dir, cfg.i_t_dir)
    i_s_dir = os.path.join(cfg.data_dir, cfg.i_s_dir)
    t_sk_dir = os.path.join(cfg.data_dir, cfg.t_sk_dir)
    t_t_dir = os.path.join(cfg.data_dir, cfg.t_t_dir)
    t_b_dir = os.path.join(cfg.data_dir, cfg.t_b_dir)
    t_f_dir = os.path.join(cfg.data_dir, cfg.t_f_dir)
    mask_t_dir = os.path.join(cfg.data_dir, cfg.mask_t_dir)
    full_image_dir = os.path.join(cfg.data_dir, cfg.full_image_dir)
    
    makedirs(i_t_dir)
    makedirs(i_s_dir)
    makedirs(t_sk_dir)
    makedirs(t_t_dir)
    makedirs(t_b_dir)
    makedirs(t_f_dir)
    makedirs(mask_t_dir)
    makedirs(full_image_dir)


    fileRecord = open("generatedatasets_140fonts_updata_hull.txt", "a", encoding="utf-8")
    # calculate the len of creating images number to give a name to image   eg: 000001  1-99999 total is 100000
    digit_num = len(str(cfg.sample_num)) - 1
    # for idx in range(cfg.sample_num):
    # classNumberList = [[0,16000],[16000,32000],[32000,48000],[48000,64000],[64000,80000],[80000,100000]]
    # classRateList=[[1,1],[0,0.25],[0.25,0.5],[0.5,0.75],[0.75,1]]
    classRate = 0
    # for classNumber,classRate in zip(classNumberList,classRateList):
    #     print("classNumber",classNumber,"classRate",classRate)
    for idx in range(80000,100000):
        print ("Generating step {:>6d} / {:>6d}".format(idx + 1, cfg.sample_num))

        # i_t, i_s, t_sk, t_t, t_b, t_f, mask_t,full_image = mp_gen.dequeue_data()
        i_s_generate = True
        try:
            # a = a1()  # 调用接口(这里把函数a1看做一个接口)
            # if a == "请求超时":
            #     print("已经超时")
            #     continue
            # elif a == "chuxiancuowu":
            #     print("chuxiancuowu")
            #     continue
            i_s,full_image,t_t ,mask_t,text,img_hull,font = datagener.gen_srnet_data_with_background(classRate)
            # print("00000000000000000000")
            # i_s, t_t, mask_t, text = a
        except:
            while i_s_generate:
                try:
                    i_s, full_image, t_t, mask_t, text,img_hull,font = datagener.gen_srnet_data_with_background(classRate)
                    i_s_generate = False
                except:
                    continue
        # # i_s,full_image,t_t ,mask_t,text,img_hull = datagener.gen_srnet_data_with_background()
        # i_s, full_image, t_t, mask_t, text, img_hull, font = datagener.gen_srnet_data_with_background(classRate)
        i_s = cv2.resize(i_s, (256, 64))
        t_t = cv2.resize(t_t, (256, 64))
        mask_t = cv2.resize(mask_t, (256, 64))
        full_image = cv2.resize(full_image, (256, 64))
        img_hull = cv2.resize(img_hull, (256, 64))
        fileRecord.write(str(idx).zfill(digit_num) + '.png' + "    " + str(text)+"    " +os.path.basename(font)+ "\n")
        i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
        t_t_path = os.path.join(t_t_dir, str(idx).zfill(digit_num) + '.png')
        full_image_path = os.path.join(full_image_dir, str(idx).zfill(digit_num) + '.png')
        mask_t_path = os.path.join(cfg.data_dir, cfg.mask_t_dir, str(idx).zfill(digit_num) + '.png')
        t_b_path = os.path.join(t_b_dir, str(idx).zfill(digit_num) + '.png')
        cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_t_path, t_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_t_path, mask_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(full_image_path, full_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_b_path, img_hull, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        # i_t, i_s, t_sk, t_t, t_b, t_f, mask_t, text = datagener.gen_srnet_data_with_background()
        # i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
        # cv2.imwrite(i_s_path, i_s,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # mask_t_path = os.path.join(cfg.data_dir, cfg.mask_t_dir, str(idx).zfill(digit_num) + '.png')
        # cv2.imwrite(mask_t_path, mask_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #
        # fileRecord.write(str(idx).zfill(digit_num) + '.png'+"    "+str(text)+ "\n")
        #
        # # i_t_path = os.path.join(i_t_dir, str(idx).zfill(digit_num) + '.png')
        # i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
        # # t_sk_path = os.path.join(t_sk_dir, str(idx).zfill(digit_num) + '.png')
        # t_t_path = os.path.join(t_t_dir, str(idx).zfill(digit_num) + '.png')
        # # t_b_path = os.path.join(t_b_dir, str(idx).zfill(digit_num) + '.png')
        # # t_f_path = os.path.join(t_f_dir, str(idx).zfill(digit_num) + '.png')
        # mask_t_path = os.path.join(cfg.data_dir, cfg.mask_t_dir, str(idx).zfill(digit_num) + '.png')
        # # full_image_path = os.path.join(full_image_dir, str(idx).zfill(digit_num) + '.png')
        #
        # # print("i_t_path",i_t[0].shape)
        # # cv2.imwrite(i_t_path, i_t[0], [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # # cv2.imwrite(t_sk_path, t_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # cv2.imwrite(t_t_path, t_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # # cv2.imwrite(t_b_path, t_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # # cv2.imwrite(t_f_path, t_f, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # cv2.imwrite(mask_t_path, mask_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # # cv2.imwrite(full_image_path, full_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])







if __name__ == '__main__':
    main()
