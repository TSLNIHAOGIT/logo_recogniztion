import cv2

img = cv2.imread('counter_test2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

_,contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

draw_img0 = cv2.drawContours(img.copy(),contours,0,(0,255,255),3)
draw_img1 = cv2.drawContours(img.copy(),contours,1,(255,0,255),3)
draw_img2 = cv2.drawContours(img.copy(),contours,2,(255,255,0),3)
draw_img3 = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)


print ("contours:类型：",type(contours))
print ("第0 个contours:",type(contours[0]))
print ("contours 数量：",len(contours))

print ("contours[0]点的个数：",len(contours[0]))
print ("contours[1]点的个数：",len(contours[1]))

# cv2.imshow("img", img)
# cv2.imshow("draw_img0", draw_img0)
# cv2.imshow("draw_img1", draw_img1)
# cv2.imshow("draw_img2", draw_img2)
# cv2.imshow("draw_img3", draw_img3)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
x, y, w, h = cv2.boundingRect(img)   
    参数：
    img  是一个二值图
    x，y 是矩阵左上点的坐标，
    w，h 是矩阵的宽和高

cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    img：       原图
    (x，y）：   矩阵的左上点坐标
    (x+w，y+h)：是矩阵的右下点坐标
    (0,255,0)： 是画线对应的rgb颜色
    2：         线宽
"""
import os
for i in range(0,len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.rectangle(img, (x,y), (x+w,y+h), (153,153,0), 5)
    new_image=img[y+2:y+h-2,x+2:x+w-2]    # 先用y确定高，再用x确定宽
    cv2.imwrite( "{}.jpg".format(i), new_image)


