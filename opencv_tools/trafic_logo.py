import cv2
import numpy as np

def load_image(path,show=True):#
    #加载原图
    img=cv2.imread(path)#
    print('img:',type(img),img.shape,img.dtype)
    if show:
        cv2.imshow('img',img)
    return img
def colours_filter_three_channels(img,show=True):
    b, _, _ = cv2.split(img)
    _, BlueThresh = cv2.threshold(b, 200, 255, cv2.THRESH_BINARY)
    if show:
        cv2.imshow("BlueTresh", BlueThresh)
    return BlueThresh
def convert_hsv(img,show=True):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    if show:
     cv2.imshow('hsv',hsv)
    return hsv
def colours_filter(hsv,show=True):
    #提取蓝色区域#提取的变为白底
    blue_lower=np.array([100,50,50])#[100,50,50]
    blue_upper=np.array([124,255,255])#124,255,255
    mask=cv2.inRange(hsv,blue_lower,blue_upper)
    print('mask',type(mask),mask.shape)
    if show:
        cv2.imshow('mask',mask)
    return mask
def fuzzy_processing(mask,show=True):
    #模糊
    blurred=cv2.blur(mask,(9,9))
    if show:
        cv2.imshow('blurred',blurred)
    return blurred
def binary_processing(blurred,show=True):
    #二值化
    ret,binary=cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
    if show:
        cv2.imshow('blurred binary',binary)
    return binary
def closed_operation(binary,show=True):
    #第一个参数控制行，第二个控制列：（参数小一点就分开了，大就合起来）
    #使区域闭合无空隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if show:
        cv2.imshow('closed',closed)
    return closed

def erode_dilate_operation(closed,show=True):
    #腐蚀和膨胀
    '''
    腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
    而膨胀操作将使剩余的白色像素扩张并重新增长回去。
    '''
    erode=cv2.erode(closed,None,iterations=4)

    dilate=cv2.dilate(erode,None,iterations=4)
    if show:
        cv2.imshow('erode', erode)
        cv2.imshow('dilate',dilate)
    return dilate

def find_contours_crop(dilate,img,show=True):
    # 查找轮廓
    image,contours, hierarchy=cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print('轮廓个数：',len(contours))
    i=0
    res=img.copy()
    for con in contours:
        #轮廓转换为矩形
        rect=cv2.minAreaRect(con)
        #矩形转换为box
        box=np.int0(cv2.boxPoints(rect))
        #在原图画出目标区域
        cv2.drawContours(res,[box],-1,(0,0,255),2)
        print([box])
        #计算矩形的行列
        h1=max([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1])
        h2=min([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1])
        l1=max([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])
        l2=min([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])
        print('h1',h1)
        print('h2',h2)
        print('l1',l1)
        print('l2',l2)
        #加上防错处理，确保裁剪区域无异常
        if h1-h2>0 and l1-l2>0:
            #裁剪矩形区域
            temp=img[h2:h1,l2:l1]
            i=i+1
            if False:
                #显示裁剪后的标志
                cv2.imshow('sign'+str(i),temp)
    #显示画了标志的原图
    if show:
        cv2.imshow('res',res)


if __name__=='__main__':
    path='traffic_test00.jpg'
    img0=load_image(path,show=False)

    # img=colours_filter_three_channels(img0)
    img=convert_hsv(img0,show=False)
    img=colours_filter(img,show=True)

    img=fuzzy_processing(img,show=False)
    img=binary_processing(img,show=True)
    img=closed_operation(img,show=True)
    img=erode_dilate_operation(img,show=True)
    img=find_contours_crop(img,img0,show=True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

