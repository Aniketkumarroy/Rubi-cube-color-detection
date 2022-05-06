import cv2 as cv
import numpy as np
from math import fabs
def dist(p1,p2):
    return (np.dot(p1-p2,p1-p2))**(0.5)
def extract(img):
    mask=np.zeros(img.shape[0:-1],np.uint8)
    img=cv.GaussianBlur(img,(5,5),0)
    img=cv.bilateralFilter(img,15,75,75)
    img=cv.fastNlMeansDenoisingColored(img, None, 5, 10, 7, 21)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    thresh=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,21,5)
    thresh=cv.erode(thresh,(3,3),iterations=2)
    img=cv.bitwise_and(img,img,mask=thresh)
    img2=img.copy()
    img=cv.Canny(img,20,150)
    img=cv.dilate(img,(7,7))
    conts,_=cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for cnts in conts:
        peri=cv.arcLength(cnts,True)
        approx=cv.approxPolyDP(cnts,0.05*peri,True)
        A,B,C,D=approx[approx[:,:,1].argmin()][0],approx[approx[:,:,0].argmax()][0],approx[approx[:,:,1].argmax()][0],approx[approx[:,:,0].argmin()][0]
        if fabs(dist(A,B)-dist(B,C))<0.04*peri and fabs(dist(B,C)-dist(C,D))<0.04*peri and fabs(dist(C,D)-dist(D,A))<0.04*peri and fabs(dist(A,C)-dist(B,D))<0.04*peri:
            cv.fillPoly(mask,[approx],(255))
    mask=cv.erode(mask,(7,7),iterations=3)
    return (mask,img2)
def redraw(img):
    IMAGE = np.zeros(img.shape,np.uint8)
    v=[[[83,131,80],[136,255,255]],                       #blue
    [[37,140,100],[82,255,255]],                          #green
    [[171,150,140],[179,255,255],[0,131,128],[5,255,255]],#red
    [[0,0,175],[179,25,255]],                             #white
    [[6,131,170],[23,255,255]],                           #orange
    [[24,115,160],[37,255,255],[38,95,195],[40,140,255]]] #yellow

    c=[(255,0,0), # blue
    (0,255,0),    #green
    (0,0,255),    #red
    (255,255,255),#white
    (0,145,255),  #orange
    (0,255,255)]  #yellow

    imghsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    for i,j in enumerate(v):
        blank=np.zeros(img.shape[:-1],np.uint8)
        lower=np.array(j[0])
        upper=np.array(j[1])
        mask=cv.inRange(imghsv,lower,upper)
        if len(j) == 4:
            lower=np.array(j[2])
            upper=np.array(j[3])
            mask2=cv.inRange(imghsv,lower,upper)
            mask = cv.add(mask,mask2)
        mask=cv.dilate(mask,(7,7),iterations=5)
        cont,_=cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        for cnts in cont:
            cv.fillPoly(blank,[cnts],(255))
        # IMG = np.zeros(img.shape,np.uint8)
        # IMG[:,:,0],IMG[:,:,1],IMG[:,:,2] = c[i][0],c[i][1],c[i][2]
        # cv.imshow("blank",blank)
        # cv.imshow("colour",IMG)
        # cv.imshow("original",img)
        # cv.waitKey(1000)
        cont,_=cv.findContours(blank,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        if len(cont) != 0:
            max_area = max([cv.contourArea(c) for c in cont])
        for cnts in cont:
            peri=cv.arcLength(cnts,True)
            approx=cv.approxPolyDP(cnts,0.05*peri,True)
            # A,B,C,D=approx[approx[:,:,1].argmin()][0],approx[approx[:,:,0].argmax()][0],approx[approx[:,:,1].argmax()][0],approx[approx[:,:,0].argmin()][0]
            # if fabs(dist(A,B)-dist(B,C))<0.1*peri and fabs(dist(B,C)-dist(C,D))<0.1*peri and fabs(dist(C,D)-dist(D,A))<0.1*peri and fabs(dist(A,C)-dist(B,D))<0.1*peri:
            if len(approx)==4 and cv.contourArea(approx) > 0.6*max_area:
                cv.fillPoly(IMAGE,[cnts],c[i])
            # cv.imshow("original",img)
            # cv.imshow("out",IMAGE)
            # cv.waitKey(1000)
    return IMAGE
def align(img):
    h,w=img.shape[0:-1]
    arr=np.array([[w//2,h//2]])
    thresh=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _,thresh=cv.threshold(thresh,5,255,cv.THRESH_BINARY)
    cont,_=cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    for cnt in cont:
        arr=np.append(arr,cnt[:,0],axis=0)
    A,B,C,D=arr[arr[:,1].argmin()],arr[arr[:,0].argmax()],arr[arr[:,1].argmax()],arr[arr[:,0].argmin()]
    pt1=np.float32([A,B,C,D])
    pt2=np.float32([[w,0],[w,h],[0,h],[0,0]])
    mat=cv.getPerspectiveTransform(pt1,pt2)
    img2=cv.warpPerspective(img,mat,(w,h))
    return img2
for i in [1,2,3,8,9,10]:
    img=cv.imread(f"input//cube{i}.jpeg")
    mask,_=extract(img)
    out=cv.bitwise_and(img,img,mask=mask)
    out=cv.fastNlMeansDenoisingColored(out,None, 30,9,6, 21)
    out=cv.bilateralFilter(out,15,75,75)
    img2=redraw(out)
    img3=align(img2)
    # cv.imshow("original",img)
    # cv.imshow("output",mask)  
    cv.imshow("redrawn",img2)
    # cv.imshow("align",img3)
    cv.waitKey(200)
cv.destroyAllWindows()
