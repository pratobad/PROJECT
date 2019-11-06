
import curses 
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

menu= ['FILTERS','SMOOTHING_BLUR','CARTOONING','GEOMETRIC_ROTATIONS','FOREGROUND_EXTRACTION','BLENDING_IMAGES','EXIT']

class Cartoonizer:
   
    def __init__(self):
        pass

    def render(self, img_rgb):
        img_rgb = cv2.imread(r'C:\Users\notme\Downloads\sky.png')
        img_rgb = cv2.resize(img_rgb, (1366,768))
        numDownSamples = 2       # number of downscaling steps
        numBilateralFilters = 50  # number of bilateral filtering steps

       
        img_color = img_rgb
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)
        
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)
        
        
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)
        
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
        (x,y,z) = img_color.shape
        img_edge = cv2.resize(img_edge,(y,x)) 
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        cv2.imwrite("edge.png",img_edge)
        return cv2.bitwise_and(img_color, img_edge)



def BLENDING_IMAGES():
    img1 = cv2.imread(r'C:\Users\notme\Downloads\sky.png')
    img2 = cv2.imread(r'C:\Users\notme\Downloads\ummmm.png')
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    dst=cv2.addWeighted(img1,0.6,img2,0.4,0)
    
    cv2.imshow('dst',dst)
    cv2.waitKey(50000)
    cv2.destroyAllWindows()
    
    
        

def FILTERS():
     
    
    def nothing(x):
        pass
    # create a black image
    img = cv2.imread(r'C:\Users\notme\Downloads\sky.png')
    cv2.imshow('image',img)
    cv2.namedWindow('image')
    
    #trackbars
    cv2.createTrackbar('R','image',0,255,nothing)
    cv2.createTrackbar('G', 'image',0,255,nothing)
    cv2.createTrackbar('B','image',0,255,nothing)
    
    #switch
    switch='0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image',0,1,nothing)
    
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(50000) & 0xFF
        if k == 27:
            break
        # current positions
        r = cv2.getTrackbarPos('R','image')
        g = cv2.getTrackbarPos('G','image')
        b = cv2.getTrackbarPos('B','image')
        s = cv2.getTrackbarPos(switch,'image')
        
        if s == 0:
            img[:] = 0
        else:
            img[:] = [b,g,r]
            
    

    cv2.destroyAllWindows()
    
def FOREGROUND_EXTRACTION():  

    img = cv2.imread(r'C:\Users\notme\Downloads\sky.png')
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    plt.imshow(img),plt.colorbar(),plt.show()    
    
   
def SMOOTHING_BLUR():
    image = cv2.imread(r'C:\Users\notme\Downloads\sky.png')
    bilateral = cv2.bilateralFilter(image,9,75,75)
    cv2.imshow('Bilateral Bluring', bilateral)
    cv2.waitKey(50000)
    cv2.destroyAllWindows()
    
    
def ROTATION():
      img = cv2.imread(r'C:\Users\notme\Downloads\sky.png')
      rows,cols,x = img.shape

      l = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
      dst = cv2.warpAffine(img,l,(cols,rows)) 
      cv2.imshow('dst',dst)
    
    
def print_menu(stdscr , selected_row_idx):
    stdscr.clear()
    h ,w = stdscr.getmaxyx()
    
    
    for idx,row in enumerate(menu):
        x=w//2 - len(row)//2
        y=h//2 - len(menu)//2 + idx
        if idx == selected_row_idx:
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(y , x , row)
            stdscr.attroff(curses.color_pair(1))
        else:
            stdscr.addstr(y , x , row)
    
    stdscr.refresh()        


def main(stdscr):
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_RED)
    
    current_row_idx = 0
    
    print(menu , current_row_idx)
    
    while 1:
        key = stdscr.getch()
        stdscr.clear()
        
        
        if key == curses.KEY_UP and current_row_idx > 0:
            current_row_idx -= 1
        elif key == curses.KEY_DOWN and current_row_idx < len(menu)-1:
            current_row_idx += 1
        elif key == curses.KEY_ENTER or key in [10,13]:
            #stdscr.addstr(0 , 0 , "You Pressed {}".format(menu[current_row_idx]))
            if menu[current_row_idx] == 'BLENDING_IMAGES':
                BLENDING_IMAGES()
                stdscr.refresh()                
                
            elif menu[current_row_idx] == 'SMOOTHING_BLUR':
                SMOOTHING_BLUR()
                stdscr.refresh()
                
            elif menu[current_row_idx] == 'FILTERS':
                FILTERS()
                stdscr.refresh()
                
            
            elif menu[current_row_idx] == 'GEOMETRIC_ROTATIONS':
                ROTATION()
                stdscr.refresh()
                
            
            elif menu[current_row_idx] == 'FOREGROUND_EXTRACTION':
                FOREGROUND_EXTRACTION()
                stdscr.refresh()
                
                
            elif menu[current_row_idx] == 'CARTOONING':
                tmp_canvas = Cartoonizer()
                file_name = "Screenshot.png" #File_name will come here
                res = tmp_canvas.render(file_name)
                cv2.imwrite("Cartoon version.jpg", res)
                cv2.imshow("Cartoon version", res)
                cv2.waitKey(50000)
                cv2.destroyAllWindows()      
            
            elif current_row_idx == len(menu) - 1:
                 break
            
            
        print_menu(stdscr , current_row_idx)
        
            
        stdscr.refresh()
        
curses.wrapper(main)      


