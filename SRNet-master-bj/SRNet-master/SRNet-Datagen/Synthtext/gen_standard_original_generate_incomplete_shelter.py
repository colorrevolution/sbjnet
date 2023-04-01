import cv2
import numpy as np
import os
import random



#  生成凸包
def generate_poly(image, n):
    """
    随机生成凸包
    :param image: 二值图
    :param n: 顶点个数
    :param area_thresh: 删除小于此面积阈值的凸包
    :return: 凸包图
    """
    # print("image",np.max(image))
    row, col = np.where(image[:, :] == 255)  # 行,列
    # print("row,col",len(row),"        ",len(col))
    point_set = np.zeros((n, 1, 2), dtype=int)
    for j in range(n):
        index = np.random.randint(0, len(row))
        point_set[j, 0, 0] = col[index]
        point_set[j, 0, 1] = row[index]
    hull = []

    hull.append(cv2.convexHull(point_set, False))
    drawing_board = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(drawing_board, hull, -1, (255, 255, 255), -1)
    # cv2.namedWindow('drawing_board', 0)
    # cv2.imshow('drawing_board', drawing_board)
    # cv2.waitKey()

    # 如果生成面积过小，重新生成
    # if cv2.contourArea(hull[0]) < area_thresh:
    #     print(cv2.contourArea(hull[0]))
    #     drawing_board = generate_poly(image, n, area_thresh)

    # # 如果生成洞，重新生成
    # is_hole = image[drawing_board == 255] == 255
    # if is_hole.all() == True:  # 洞，则drawing_board所有为255的地方，image也是255，all()即为所有位置
    #     drawing_board = generate_poly(image, n, area_thresh)
    return drawing_board

# 构建字符遮挡
def add_shelter(image,multi,shelter_rate_section):
    # print("image",image.shape)
    # print("multi",multi)
    # 生成30%的连续字符
    if np.random.random_sample() < 0.3:
        length_multi = len(multi)
        # 确定合并的数量
        if length_multi >=2:
            merge_number = np.random.randint(2,length_multi+1)
            # 确定起始合并位置
            if length_multi == merge_number:
                start_location = 0
            else:
                start_location = np.random.randint( length_multi - merge_number )
            merge_multi = multi[start_location:start_location+merge_number]
            merge_multi = np.array(merge_multi)
            left = merge_multi[0][0]
            right = merge_multi[-1][1]
            up = np.min(merge_multi[:,2])
            bottom = np.max(merge_multi[:,3])
            # 存放临时性的合并列表
            temporary_list = []
            temporary_list.append(left)
            temporary_list.append(right)
            temporary_list.append(up)
            temporary_list.append(bottom)
            del_index = range(start_location,start_location+merge_number)
            multi = [multi[i] for i in range(len(multi)) if (i not in del_index)]
            multi.append(temporary_list)
            multi = sorted(multi,key=(lambda x: x[0]))


    # 随机生成遮挡字符的比例
    shelter_character_rate = np.random.uniform(shelter_rate_section[0],shelter_rate_section[1])

    # print("len",len(multi))
    # 计算需要遮挡的字母数量
    shelter_character_number = int(shelter_character_rate * len(multi))
    if shelter_character_number == 0 :
        shelter_character_number = 1

    # 随机选择shelter_character_number个需要遮挡的字符的索引
    character_index = random.sample(range(0,len(multi)),shelter_character_number)
    # # 排序
    # character_index = sorted(character_index)
    # print("character_index",character_index)
    multi_array = np.array(multi)
    new_multi = multi_array[character_index]
    # print("shelter_character_number",shelter_character_number,len(multi),shelter_character_rate,new_multi)

    # save img_hull
    save_img_hull = []
    full_img_hull = np.zeros(image.shape, dtype=np.uint8)
    # print("multi[i]", multi)
    for i in range(len(new_multi)):
        # 取出数据
        left = multi[i][0]
        right = multi[i][1]
        up = multi[i][2]
        bottom = multi[i][3]
        # print("multi[i]",multi[i])

        # 判断宽度是否等于1,如果等于1,就进行扩展到三个像素
        if right - left < 3:
            right = right+3
        if bottom -up <3:
            bottom = bottom +3

        # 创建画布
        height = bottom - up
        width = right - left
        # print("height",height)
        # 计算画布的面积
        area_sub_image = height * width

        img = np.zeros((height, width), np.uint8)
        # print("img",img.shape)
        value = min(height, width)

        #将半径从四分之一开始改为从0开始,半径最大设置为款或者高
        # value_1_2 = int(value / 2)
        # value_1_4 = int(value / 4)
        value_1_2 = value
        value_1_4 = 0

        # print("value",value,value_1_2,value_1_4)

        # 遮挡形状选择的概率
        c_p = np.random.random_sample()
        if c_p <= 0.45:
            # print("圆的凸包")
            # 圆的凸包
            # 半径
            radius = np.random.randint(value_1_4, value_1_2)
            # 原点的位置
            center_x = np.random.randint(width)
            center_y = np.random.randint(height)

            # center_x = np.random.randint(radius, width - radius)
            # center_y = np.random.randint(radius, height - radius)
            cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), -1)
            # 顶点个数
            vertex_number = np.random.randint(8, 60)
            # # 面积阈值
            # area_thresh = np.random.randint(0, int(area_sub_image * 3 / 4))
            #
            # 查看字符区域中字符所占的像素数量
            character_pix_number_original = np.sum(image[up:bottom, left:right]) / 255
            # 判断遮挡面积是否满足  剩余字符面积要大于百分之二十,或者迭代50轮终止
            is_satisfy = False
            a = 0
            while not is_satisfy:
                a = a + 1
                img_hull = generate_poly(img, vertex_number)
                img_hull = 1 - (img_hull / 255)
                # 查看字符区域中字符被遮挡后所占的像素数量
                # print("image[up:bottom, left:right]", image[up:bottom, left:right].shape, img_hull.shape)
                character_pix_number_shelter = np.sum(image[up:bottom, left:right] * img_hull) / 255

                if character_pix_number_shelter / character_pix_number_original > 0.2:
                    is_satisfy = True
                if a > 50:
                    break
            image[up:bottom, left:right] = image[up:bottom, left:right] * img_hull
            # jiang xiao kuai tian chong dao da tu zhong
            full_img_hull[up:bottom, left:right] = (1-img_hull) * 255
        elif 0.45 < c_p <= 0.8:
            # print("矩形的凸包")
            # 矩形的凸包
            left_rectangle = np.random.randint(0, width)
            right_rectangle = np.random.randint(left_rectangle, width)
            up_rectangle = np.random.randint(0, height)
            bottom_rectangle = np.random.randint(up_rectangle, height)
            cv2.rectangle(img, (left_rectangle, up_rectangle), (right_rectangle, bottom_rectangle), (255, 255, 255), -1)
            # 顶点个数
            vertex_number = np.random.randint(8, 30)
            # 面积阈值
            # 查看字符区域中字符所占的像素数量
            character_pix_number_original = np.sum(image[up:bottom, left:right]) / 255
            # 判断遮挡面积是否满足
            is_satisfy = False
            b = 0
            while not is_satisfy:
                b = b + 1
                img_hull = generate_poly(img, vertex_number)
                img_hull = 1 - (img_hull / 255)
                # 查看字符区域中字符被遮挡后所占的像素数量
                # print("image[up:bottom, left:right]", image[up:bottom, left:right].shape, img_hull.shape)
                character_pix_number_shelter = np.sum(image[up:bottom, left:right] * img_hull) / 255
                if character_pix_number_shelter / character_pix_number_original > 0.2:
                    is_satisfy = True
                if b > 50:
                    break
            image[up:bottom, left:right] = image[up:bottom, left:right] * img_hull
            # jiang xiao kuai tian chong dao da tu zhong
            full_img_hull[up:bottom, left:right] = (1 - img_hull) * 255
        elif 0.8 < c_p <= 1:
            # print("矩形")
            # 查看字符区域中字符所占的像素数量
            character_pix_number_original = np.sum(image[up:bottom, left:right]) / 255
            # 判断遮挡面积是否满足
            is_satisfy = False
            c = 0
            while not is_satisfy:
                c = c + 1
                # 矩形
                left_rectangle = np.random.randint(0, width)
                right_rectangle = np.random.randint(left_rectangle, width)
                up_rectangle = np.random.randint(0, height)
                bottom_rectangle = np.random.randint(up_rectangle, height)
                cv2.rectangle(img, (left_rectangle, up_rectangle), (right_rectangle, bottom_rectangle), (255, 255, 255),-1)
                img_hull = img
                img_hull = 1 - (img_hull / 255)
                # 查看字符区域中字符被遮挡后所占的像素数量
                # print("image[up:bottom, left:right]", image[up:bottom, left:right].shape, img_hull.shape)
                character_pix_number_shelter = np.sum(image[up:bottom, left:right] * img_hull) / 255
                if character_pix_number_shelter / character_pix_number_original > 0.2:
                    is_satisfy = True
                if c > 50:
                    break
            image[up:bottom, left:right] = image[up:bottom, left:right] * img_hull
            # jiang xiao kuai tian chong dao da tu zhong
            full_img_hull[up:bottom, left:right] = (1 - img_hull) * 255
    return  image ,full_img_hull




# 遮挡字符的数量占总体字符数量的区间：
# shelter_rate_section = [0,0.25]
# shelter_rate_section = [0.75,1]

# multi = []
# single = []
#
# # txt 文件的路径
# file_txt = open('character_coordinate.txt', 'a')
# # mask 图像的路径
# filePath = "mask_image"
#
# imageList = os.listdir(filePath)
# for name in imageList:
#     image = cv2.imread(os.path.join(filePath,name))
#     image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
def main_shelter(image,bbs,classRate):

    imgH, imgW = image.shape
    # 左右上下
    multi = [[box[0],box[0]+box[2],box[1],box[1]+box[3]] for box in bbs]
    height,width = image.shape[0:2]
    # print("height,width",height,width)
    # print("multi",multi)
    for value in multi[::-1]:
        if value[0]<0 or value[1]<0 or value[2]<0 or value[3]<0:
            multi.remove(value)
        elif value[1]>width or value[3]>height:
            multi.remove(value)


    #自动选择字符遮挡区域

    shelter_image,save_img_hull = add_shelter(image,multi,classRate)

    return shelter_image , save_img_hull


if __name__ == '__main__':

    height=51
    width = 248
    multi = [[24, 43, 29, 65], [46, 61, 5, 76], [64, 83, 29, 76], [86, 111, 28, 78], [114, 128, 19, 42], [122, 152, 11, 42], [164, 180, 28, 42]]
    for value in multi[::-1]:
        if value[0] < 0 or value[1] < 0 or value[2] < 2 or value[3] < 0:
            multi.remove(value)
        elif value[1] > width or value[3] > height:
            multi.remove(value)
    print(multi)