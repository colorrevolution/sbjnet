import  os
import pickle as cp
import numpy as np
from pygame import freetype
# from . import data_cfg
font_list = os.listdir(r"/media/mmsys9/系统/xff/TextErase/SRNet-master/SRNet-master/datasets/fonts/english_ttf/")
print(type(font_list))
# font = np.random.choice(font_list,10)
# print(font)





current_file_path = os.path.dirname(__file__)
textPath = os.path.join(current_file_path,"data/texts.txt")
textlist = open(textPath,"r").readlines()
# print(textlist)
textlist = [text.strip() for text in textlist]

# print(textlist)

bg_filepath = '/media/mmsys9/系统/xff/TextErase/SRNet-master/SRNet-master/datasets/imnames.cp'

with open(bg_filepath, 'rb') as f:
    bg_list = set(cp.load(f))
    bg_list = ['/media/mmsys9/系统/xff/TextErase/SRNet-master/SRNet-master/datasets/bg_data/bg_img/'+img_path.strip() for img_path in bg_list]
    # bg_list_1 = [img_path.strip() for img_path in bg_list]


# print(bg_list)
#
freetype.init()
font_list = os.listdir("/media/mmsys9/系统/xff/TextErase/SRNet-master/SRNet-master/datasets/fonts/english_ttf/") # retern the list of file or folder
font_list = [os.path.join("/media/mmsys9/系统/xff/TextErase/SRNet-master/SRNet-master/datasets/fonts/english_ttf/", font_name) for font_name in font_list]
font = np.random.choice(font_list)
print("font",font)
font = freetype.Font(font)
# init font
# font = freetype.Font(font)
font.antialiased = True
font.origin = True

# choose font style
font.size = np.random.randint(25, 61)
font.underline = np.random.rand() < 0.01
font.strong = np.random.rand() < 0.07
font.oblique = np.random.rand() < 0.02
#pygame.Rect( left , top, width, height )
line_bounds = font.get_rect("love11")
print(line_bounds.center,"   ",line_bounds.width,"   ",line_bounds.height)
print(line_bounds.center,"   ",line_bounds.x,"   ",line_bounds.y)
print(type(line_bounds),line_bounds)
text= "111233 3xddd"
text1 = "我是薛凡福"
words = ' '.join(text.split())
wor = " ".join("我是薛凡福")
wordstrip=  " ".join(text1.split())
print(words)
print(wor)
print(wordstrip)


padding_ud = np.random.randint(0, 10, 2)
padding_lr = np.random.randint(0, 20, 2)
padding = np.hstack((padding_ud, padding_lr))
print(padding)

a = np.random.randn(2)
print(a)

perspect =0.0005 * np.random.randn(2) + 0
print(perspect)



col_file = 'data/colors.cp'


import cv2
def get_color_matrix(col_file):

    with open(col_file,'rb') as f:
        colorsRGB = cp.load(f, encoding ='latin1')
    ncol = colorsRGB.shape[0]
    colorsLAB = np.r_[colorsRGB[:,0:3], colorsRGB[:,6:9]].astype(np.uint8)
    colorsLAB = np.squeeze(cv2.cvtColor(colorsLAB[None,:,:], cv2.COLOR_RGB2Lab))
    return colorsRGB, colorsLAB

colorsRGB, colorsLAB = get_color_matrix(col_file)
print("colorsLAB[0]",colorsLAB[0])
print("colorsRGB[0]",colorsRGB[0])
print(colorsRGB.shape)
print(colorsLAB.shape)

a = tuple(np.random.randint(0, 256, 3))
print(a)

x = [5, 6, 2, 1, 6, 7, 2, 7, 9]
x.insert(3,888)
print(x)

x= [[1,2],[2,3]]
x_np = np.array(x)
print(x_np.shape)
print(x_np[None,:].shape)
x= np.mean(x_np,0)
print("eee",x.shape)

x= np.random.randint(0,256,(4,5,3))
x = np.mean(x,1)
x = np.mean(x,1)
# x = np.mean(x,0)
print(x)
print(x.shape)

x = [5, 6, 2, 1, 6, 7, 2, 7, 9]
x7= [[1,2],[2,3],[1,2]]


x7 = np.array(x7)
print("-----------------",x7.size)
# print(x.shape)
# x = np.atleast_2d(x)
# print(x)
# print(x.shape)

bg = np.ones((200 , 200)) * 255
print(bg.shape)