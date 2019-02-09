import cv2
import numpy as np
import math
import sys

from skimage.exposure import rescale_intensity

args = sys.argv
imgOrg =  np.zeros((5,5), np.uint8)
imgMod =  np.zeros((5,5), np.uint8)
cam = 0


def run():
    print("Please insert a command. Press 'h' to see the options")
    value = raw_input("prompt: ")
    value_list = value.split()
    if((len(value_list)==1) and value_list[0]=='load'):
        getImageFromCamera()
    if(len(value_list)==2):
        getImage(value_list[1])
    
    while(1):
         #if (value.split()[1] != "notaname"):
          #   getImage(value.split()[1])
         #if value.split()[0] == 'load':
           #  getImageFromCamera()
         if value == 'i':
             print 'Estoy en i'
             i()
         elif value == 'w':
             w()
         elif value == 'g':
             print 'Estoy en g'
             g()   
         elif value == 'G':
             G()
         elif value == 'c':
             c()
         elif value == 's':
             s()
         elif value == 'S':
             S()
         elif value == 'd':
             d()
         elif value == 'D':
             D()
         elif value == 'x':
             x()
         elif value == 'y':
             y()
         elif value == 'm':
             m()
         elif value == 'p':
             p()
         elif value == 'r':
             r()
         elif value == 'h':
             h()
         if value == 'e':
             break
         
         
def getImageFromCamera():
    global cam, imgOrg
    cam = 1
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("image")
    #img_counter = 0
    
    while(1):
        ret,frame = cam.read()
        cv2.imshow("image",frame)
        if not ret:
            break
        imgOrg = frame
        value = cv2.waitKey(50)&0xff
        if(value == ord('q')):
            cv2.destroyAllWindows()
            run()
            break
    

def getImage(imageName):
    
    global imgOrg
    imgOrg = cv2.imread(imageName)
    cv2.imshow('image',imgOrg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    run()
    
# Reload the original image: Cancel any previous processing
def i():
    global imgMod, imgOrg
    imgMod = imgOrg
    cv2.imshow('Original image', imgOrg)
    run()
    
# Save image    
def w ():
    global imgMod, imgOrg
    cv2.imwrite("out.jpg",imgMod)
    run()
    
# Convert the image to grayscale using openCV function
def g():
    global imgMod, imgOrg
    if cam == 1:
        imgOrg = getImageFromCamera()
    imgGray = cv2.cvtColor(imgOrg, cv2.COLOR_RGB2GRAY)
    cv2.imshow('Gray scale openCV',imgGray)
    imgMod = imgGray
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    run()
       
# Convert the image to grayscale using your implementation of conversion function
def G():
    global imgMod, imgOrg
    if cam == 1:
        imgOrg = getImageFromCamera()
    
    imgGray = imgOrg
    
    R = np.array(imgGray[:,:,0])
    G = np.array(imgGray[:,:,1])
    B = np.array(imgGray[:,:,2])
    
    R = (R*.299)
    G = (G*.587)
    B = (B*.114)
    
    Avg = (R+G+B)
    for i in range(3):
        imgGray[:,:,i]=Avg
        
    cv2.imshow('Gray scale no openCV',imgGray)
    imgMod = imgGray
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    run()
    
# Cycle through the color channels of the image
# showing a different channel every time the key is pressed    
def c():
    global imgMod, imgOrg
    i = 0
    if cam == 1:
        imgOrg = getImageFromCamera()
    
    print("Press 'c' to change color")
    cv2.imshow('Original Image', imgOrg)
    
    while(1):
        value = cv2.waitKey(50)&0xff        
        if value ==ord('c') and (i == 0):
            # RGB - Blue
           b = imgOrg.copy()
           # Set green and red channels to 0
           b[:,:,1] = 0
           b[:,:,2] = 0
           
           cv2.imshow('RGB', b)
           i=1
           imgMod = b
        elif value == ord('c') and (i==1):
            # RGB - Green
            g = imgOrg.copy()
            g[:,:,0] = 0
            g[:,:,2] = 0
            
            cv2.imshow('RGB', g)
            i=2
            imgMod = g
        elif value ==ord('c') and (i==2):
           # RGB - Red
           r = imgOrg.copy()
           r[:,:,0] = 0
           r[:,:,1] = 0
           cv2.imshow('RGB', r)
           i=0
           imgMod = r
        
        if(value == ord('q')):
            cv2.destroyAllWindows()
            run()
            break
            
    
def nothing(x):
    pass  
 
# Convert image to grayscale
# Smooth image with track bar  
def s():
    global imgMod, imgOrg
    if cam == 1:
        imgOrg = getImageFromCamera()
    if(imgOrg.shape != (imgOrg.shape[0],imgOrg.shape[1])):
        imgGray = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('Smoothing',0)
    cv2.createTrackbar('s()','Smoothing',0,10,nothing)
    
    while(True):
        img = imgGray.copy()
        smooth = int(cv2.getTrackbarPos('s()','Smoothing'))+1
        
        img = cv2.blur(img,(smooth, smooth))
        cv2.imshow('Smoothing',img)
        imgMod = img
        value = cv2.waitKey(50)&0xff
        if(value == ord('q')):
            cv2.destroyAllWindows()
            run()
            break
    run()

def convolve(img,kernel):
    
    (iH, iW) = img.shape[:2]
    (kH,kW) = kernel.shape[:2]
    
    pad = (kW -1)//2
    img = cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype = "float32")
    
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            r = img[y - pad: y + pad +1, x - pad:x + pad + 1]
            k = (r*kernel).sum()
            output[y-pad,x-pad]=k
    output = rescale_intensity(output, in_range=(0,255))
    output = (output*255).astype("uint8")
    
    return output

# Convert the image to grayscale and smooth it using your function which should
# perform convolution with a suitable filter
# You need to implement your own convolution here
# Use a trackbar to control the amount of smoothing
def S():
    global imgMod, imgOrg
    if cam == 1:
        imgOrg = getImageFromCamera()
    imgGray = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('track',0)
    cv2.createTrackbar('S()','track',0,10,nothing)
    while(True):
        img = imgGray.copy()
        smooth = int(cv2.getTrackbarPos('S()','track'))+1
        kernel = np.ones((5,5),np.float32)/(smooth*smooth)
        img = convolve(img,kernel)
        cv2.imshow('track',img)
        imgMod = img
        value = cv2.waitKey(50)&0xff
        if(value == ord('q')):
            cv2.destroyAllWindows()
            run()
            break
   
#Conver the image to grayscale and smooth it using your function which should
# perform convolution with a suitable filter
# You need to implement your own convolution function here
# Use trackbar to control the amount of smoothing

# Downsample the image by a factor of 2 without smoothing
def d():
    global imgMod, imgOrg
    if cam == 1:
        imgOrg = getImageFromCamera()
    imgR = cv2.resize(imgOrg, None, fx=0.5,fy=0.5, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('Downsample', imgR)
    imgMod = imgR
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    run()

# Downsample the image by a factor of 2 with smoothing
def D():
    global imgMod, imgOrg
    if cam == 1:
        imgOrg = getImageFromCamera()
    imgS = cv2.pyrDown(imgOrg)
    cv2.imshow('Smooth and downsample',imgS)
    imgMod = imgS
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    run()
 
# Convert the image to grayscale and perform convolution with an x derivative filter.
# Normalize the obtained values to the range [0,255]
def x():
    global imgMod,imgOrg
    if cam == 1:
        imgOrg = getImageFromCamera()
    imgGray = cv2.cvtColor(imgOrg, cv2.COLOR_RGB2GRAY)
    img = cv2.Sobel(imgGray,cv2.CV_64F,1,0,ksize=5)
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)
    cv2.imshow('image',img)
    imgMod = img
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    run()

# Convert the image to grayscale and perform convolution with an x derivative filter.
# Normalize the obtained values to the range [0,255]
def y():
    global imgMod,imgOrg
    if cam == 1:
        imgOrg = getImageFromCamera()
    imgGray = cv2.cvtColor(imgOrg, cv2.COLOR_RGB2GRAY)
    img = cv2.Sobel(imgGray,cv2.CV_64F,0,1,ksize=5)
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)
    cv2.imshow('image',img)
    imgMod = img
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    run()

# Show the magnitude of the gradient normalized to the range [0,255]
# The gradient is computed based on the x and y derivatives of the image
def m():
    global imgMod,imgOrg
    if cam == 1:
        imgOrg = getImageFromCamera()
    imgGray = cv2.cvtColor(imgOrg, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(imgGray,cv2.CV_64F,1,0,ksize=5)
    dy = cv2.Sobel(imgGray,cv2.CV_64F,0,1,ksize=5)
   
    magnitude =  cv2.magnitude(dx, dy, None)
    magnitude = cv2.normalize(magnitude, None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)
    imgMod = magnitude
    cv2.imshow('magnitude',magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    run()
  
# Convert the image to grayscale and 
# plot the gradient vectors of the image every N pixels
# and let the plotted gradient vectors have a length of K
# Use a track bar to control N
# Plot the vectors as short line segments of length K
def p():
    
    global imgMod,imgOrg
    if cam == 1:
        imgOrg = getImageFromCamera()
    
    imgGray = cv2.cvtColor(imgOrg, cv2.COLOR_RGB2GRAY)
    
    cv2.namedWindow('Arrows',0)
    cv2.createTrackbar('N','Arrows',0,19,nothing)
    
    while(True):
        img = imgGray.copy()
        N = int(cv2.getTrackbarPos('N','Arrows'))+1
        dx = cv2.Sobel(imgGray,cv2.CV_64F,1,0,5)
        dy = cv2.Sobel(imgGray,cv2.CV_64F,0,1,5)
        k  = 10
        for i in range(0,img.shape[0],N):
            for j in range(0,img.shape[1],N):
                angle = math.atan2(dy[i][j],dx[i][j])
                x = int(i+k*math.cos(angle))
                y = int(j+k*math.sin(angle))
                
                cv2.arrowedLine(img,(i,j),(x,y),(0,0,0))
         
        cv2.imshow('Arrows', img)
        value = cv2.waitKey(50)&0xff
        if(value == ord('q')):
            cv2.destroyAllWindows()
            run()
            break

       
# Convert the image to grayscale and rotate it using an angle of tetha degrees
# Use a track bar to control the rotation angle
# The rotation of the image should be performed using an inverse map
# so there are no holes in it. 
# Use the cv2.getRotationMatrix2D and cv2.warpAffine functions
def r():
    global imgMod,imgOrg
    if cam == 1:
        imgOrg = getImageFromCamera()
    imgGray = cv2.cvtColor(imgOrg, cv2.COLOR_RGB2GRAY)
    cv2.namedWindow('Rotation',0)
    cv2.createTrackbar('Degrees','Rotation',0,360,nothing)
    rows, cols = imgGray.shape
    while(True):
        img = imgGray.copy()
        degrees = cv2.getTrackbarPos('Degrees','Rotation')
        temp = cv2.getRotationMatrix2D((cols/2,rows/2),degrees,1)
        img = cv2.warpAffine(img,temp,(cols,rows))
        cv2.imshow('Rotation',img)
        imgMod = img
        value = cv2.waitKey(50)&0xff
        if(value == ord('q')):
            cv2.destroyAllWindows()
            run()
            break
 
# Display a short description of the program, its command line arguments,
# and the keys it supports
def h():
    print("Program description: ")
    print("q: quit a function")
    print("load: Load an image. If no file is specified, the image will be captured from camera")
    print("i: Reload original image" )
    print("w: save the current image")
    print("g: convert the image to grayscale using the openCV conversion function ")
    print("G: convert the image to grayscale using your implementation of conversion function")
    print("c: cycle through the color channels of the image showing a different channel every time the key is pressed")
    print("s: conver the image to grayscale and smooth it using the openCV function. Trackbar to control the amount of smoothing ")
    print("S: conver the image to grayscale and smooth it using a function which should perform convolution with a suitable filter. Track bar to control smoothing ")
    print("d: Downsample the image by a factor of 2 without smoothing")
    print("D: downsample the image by a factor of 2 with smoothing")
    print("x: convert the image to grayscale and perform convolution with an x derivative filter. Normalize the obtained values to the range [0,255]")
    print("y: convert the image to grayscale and perform the convolution with a y derivative filter. Normalize the obtained values to the range [0,255] ")
    print("m: show the magnitude of the gradient normalized to the range [0,255]. The gradient is computed based in the x and y derivatives of the image")
    print("p: Convert the image to grayscale and plot the gradient vectors of the image very N pixels and let the plotted gradient vectors have a length of K. Use a track bar to control N.\n Plot the vectors as short line segments of length K ")
    print("r: Convert the image to grayscale and rotate it using an angle of tetha egrees. Use a track bar to control the rotation angle. \n The rotation of the image performed using getRotationmatrix2D and warpAffine functions")
    run()
    

    
    
    
