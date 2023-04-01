import cv2
import numpy as np
path = "black-chinese.png"
image = cv2.imread(path,0)
image = 255-image
cv2.imwrite("reverse-black-chinese.png",image)
cv2.imshow("img",image)
cv2.waitKey(0)

# # 找出文本的上下左右边界
# high,width = image.shape[0:2]
# print("high",high,width)
# for i in range(high):
#     # 确定top的值
#    if np.sum(image[i,:])>100:
#        print(i,np.sum(image[i,:]))
#        top = i
#        break
#
# for i in range(high):
#     # 确定bottom的值
#    if np.sum(image[high-1-i,:])>100:
#        bottom = high-1-i
#        break
#
# for i in range(width):
#     # 确定left的值
#    if np.sum(image[:,i])>100:
#        left = i
#        break
# for i in range(width):
#     # 确定right的值
#    if np.sum(image[:,width-1-i])>100:
#        right = width-1-i
#        break


# 找出文本的上下左右边界
high,width = image.shape[0:2]
print("high",high,width)
for i in range(high):
    # 确定top的值
   if 255 in image[i,:]:
       print(i,np.sum(image[i,:]))
       top = i
       break

for i in range(high):
    # 确定bottom的值
   if 255 in image[high-1-i,:]:
       bottom = high-1-i
       break

for i in range(width):
    # 确定left的值
   if 255 in image[:,i]:
       left = i
       break
for i in range(width):
    # 确定right的值
   if 255 in image[:,width-1-i]:
       right = width-1-i
       break



print("top",top,bottom,left,right)
image = image[top:bottom,left:right]
cv2.imshow("img",image)
cv2.waitKey(0)


