import cv2
import os
import numpy as np
from PIL import Image
import random

os.chdir(os.path.dirname(os.path.realpath(__file__)))


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])


def createBackGround():
    S = (0.5, 0.5, 0.5, 0.5)  # Define blending coefficients S and D
    D = (0.5, 0.5, 0.5, 0.5)
    skyImg = cv2.imread('src/sky.png')
    grassImg = cv2.imread('src/grass.png', -1)  # 350
    overlay = grassImg.copy()
    output = skyImg.copy()

    overlay_image_alpha(output,
                        overlay[:, :, 0:3],
                        (0, 350),
                        overlay[:, :, 3] / 255.0)

    overlay_image_alpha(output,
                        overlay[:, :, 0:3],
                        (1005, 350),
                        overlay[:, :, 3] / 255.0)

    grassImg2 = cv2.imread('src/grass2.png', -1)  # 460

    overlay = grassImg2.copy()
    overlay_image_alpha(output,
                        overlay[:, :, 0:3],
                        (0, 460),
                        overlay[:, :, 3] / 255.0)

    overlay_image_alpha(output,
                        overlay[:, :, 0:3],
                        (1005, 460),
                        overlay[:, :, 3] / 255.0)

    cv2.imwrite('background.png', output)
    # cv2.imshow('output',output)
    # cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)

    # drawing = np.zeros((skyImg.shape[0],skyImg.shape[1]*2,skyImg.shape[2]),np.uint8)
    # drawing[:,0:1024] = skyImg
    # drawing[350:402,0:1005] = grassImg
    # drawing[:,1023:2047] = skyImg
    # cv2.imshow('grass', grassImg)
    # cv2.imshow('bgImg',drawing)

    # cv2.waitKey(0)


def birdDataGen(birdType, index):
    bg = cv2.imread('src/bg_day.png')
    bg0 = bg.copy()
    bird0_0 = cv2.imread('src/bird2_0.png', -1)
    bird0_1 = cv2.imread('src/bird2_1.png', -1)
    bird0_2 = cv2.imread('src/bird2_2.png', -1)
    birds = [bird0_0, bird0_1, bird0_2]
    cv2.imshow('bird', birds[random.randint(0, 2)])
    size = 40  # random.randint(100,300)
    x = 0
    y = 1
    while x != y:
        offsetx = random.randint(-10, 10)
        offsety = random.randint(-10, 10)
        posx = random.randint(0, bg.shape[1]-50)
        posy = random.randint(0, bg.shape[0]-50)
        overlay_image_alpha(bg0, bird0_0[:, :, 0:3],
                            (posx, posy), bird0_0[:, :, 3] / 255.0)
        img50 = bg0[posy+5+offsety:posy+size+5 +
                    offsety, posx+2+offsetx:posx+size+2+offsetx]
        # ret = cv2.resize(img50, (size, size), interpolation=cv2.INTER_CUBIC)
        x, y, _ = img50.shape
    cv2.imwrite('data/positive/'+str(index)+'.bmp', img50)
    # cv2.imshow('output',bg0)
    # cv2.waitKey(0)


def birdGen(birdType):
    bg = cv2.imread('src/bg_day.png')
    bg0 = bg.copy()
    bird0_0 = cv2.imread('src/bird2_0.png', -1)
    cv2.imshow('bird', bird0_0)
    size = 50  # random.randint(100,300)
    posx = random.randint(0, bg.shape[1]-size)
    posy = random.randint(0, bg.shape[0]-size)
    overlay_image_alpha(bg0, bird0_0[:, :, 0:3],
                        (posx, posy), bird0_0[:, :, 3] / 255.0)
    cv2.imwrite('output.png', bg0)


def negDataGen(index):
    bg = cv2.imread('src/bg_night.png')
    bg0 = bg.copy()
    size = 40
    x = 0
    y = 1
    while x != y:
        posx = random.randint(0, bg.shape[1]-size)
        posy = random.randint(0, bg.shape[0]-size)
        img50 = bg0[posy:posy+size, posx:posx+size]
        x, y, _ = img50.shape
    cv2.imwrite('data/negative/'+str(index)+'.bmp', img50)


def readAllImg(path, *suffix):
    files = os.listdir(path)
    pics = []
    for file in files:
        endStr = file.split('.')[-1]
        if endStr in suffix:
            pics.append(file)
    return pics


def infoTxtPositive(num):
    with open(r'pos.txt', 'w') as writer:
        for i in range(num):
            writer.write('./data/positive/{0}.bmp 1 0 0 38 38\n'.format(i))


def infoTxtNegative(num):
    with open(r'neg.txt', 'w') as writer:
        picList = readAllImg('./data/negative', 'bmp')
        # print(picList)
        for i in picList:
            tmpimg = cv2.imread('./data/negative/{0}'.format(i), -1)
            writer.write('./data/negative/{0}\n'.format(i,
                                                   tmpimg.shape[0], tmpimg.shape[1]))


def birdRecognition():
    bird_cascade = cv2.CascadeClassifier("data/cascade.xml")
    # face_cascade.load("haarcascade_frontalcatface_extended.xml")

    img = cv2.imread('output.png')
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = bird_cascade.detectMultiScale(
        gray, 1.2, 5, cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        for faceRect in faces:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2, 8, 0)
            '''
            IDrectx = x-3*w if (x-3*w)>0 else 0
            IDrecty = int(y-0.8*h) if (y-0.8*h)>0 else 0
            IDrectw = int(x+1.6*w) if (x+1.6*w)<img.shape[1] else img.shape[1]
            IDrecth = int(y+2.3*h) if (y+2.3*h)<img.shape[0] else img.shape[0]
            cv2.rectangle(img,(IDrectx,IDrecty),(IDrectw,IDrecth),(0,255,0),2,8,0)   
            '''
    cv2.namedWindow('img', 0)
    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # createBackGround()
    # for i in range(300):
    #    birdDataGen(0, i)
    #infoTxtPositive(300)
    for i in range(500, 700):
       negDataGen(i)
    #infoTxtNegative(300)
    # birdGen(0)
    # birdRecognition()
