#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 02:50:23 2020

@author: mv96
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

def single_channel_grey(img,color=0):
    """
    1- converts the image into gray scale +2pts
    2- makes sure the data type is float 32 +1 pts
    3- resizes the image to double its size making sure we view the new size of the image +2 pts
    """
    #total 5 point function
    #this function take the image and converts the image into a greyscale image +2 points
    
    image = cv2.imread(img,color) 

    
    #showing the image to the user
    #cv2.imshow("Greyscale img",image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    print("we can verify the image is greyscale by\nlooking at the shape of the image: {}".format(image.shape))
    original_height=int(image.shape[0])
    original_width=int(image.shape[1])
    print('Height of Image:', original_height, 'pixels')
    print('Width of Image: ', original_width, 'pixels')
    
    #we can see here the image dtype is uint 8
    print('image dtype: ',image.dtype)
    print(image[0][0])
    
    print("===now converting the image's dtype to float===")
    #let's convert this image dtype to float 32 fromunit8
    image = np.float32(image)
    print('image dtype: ',image.dtype)
    print(image[0][0])
    
    #we can convert back to uint8 by using the below set of 3 simple commands 
    #image = image.astype(np.uint8)
    #print('image dtype: ',image.dtype)
    #print(image[0][0])
    
    #execute the below commands to view the image
    #cv2.imshow("Greyscale img",image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    scale_factor=2 #this is the scaling factor by default it's 2 because we need to double the image size
    dim=(int(original_width*scale_factor),int(original_height*scale_factor)) #dim is always width, height
    #resizes the image to double it's size +2 point
    #we have used interpolation ==>> lanczos4 because it's theoritically the best 
    #refer- https://stackoverflow.com/questions/3112364/how-do-i-choose-an-image-interpolation-method-emgu-opencv
    print("resizing the image........")
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_LANCZOS4)
    new_height=int(resized_image.shape[0])
    new_width=int(resized_image.shape[1])
    print('Height of Image:', new_height, 'pixels')
    print('Width of Image: ', new_width, 'pixels')
    
    #plot float32 image using matplot lib
    #refer discussion forums where christian recommends matplot lib https://cit.instructure.com/courses/29067/discussion_topics/21512
    plt.figure()
    plt.imshow(resized_image)
    
    return resized_image

    
    
    
def gaussian_kernels(img,n=12,window_size=3):
    """
    note- the image reeived from the previous function is float32 we can't view it very clearly unless when changed in different format
    
    1- generates n such gaussian kernels default is 12 +4 pts
    
    (please see the window size as it carefully fits the derivative of gaussian function
    thus we made sure we sufficiantly captured the gaussian kernel
    window size default is 5 because it fits fell for the blobs)
    
    2-apply these kernels to resized image from 1st task +2pts
    
    notice as soon as we icrease the sigma value the plot smoothens alot not capturing the fine details this indicates exactly 
    how it should suppose to be working
    """
    #A list of all possible sigma values
    
    sigma_vals=list(range(0,n))
    sigma_vals = list(map(lambda x: 2**(x/2),sigma_vals))

    filtered_images=[]
    
    for i,sigma in enumerate(sigma_vals):
        
        # make a meshgrid of x and y based on the sigma values
        x,y = np.meshgrid(np.arange(0,window_size*sigma),np.arange(0,window_size*sigma))
        
        #make a gaussian kernel
        gaussian_filter = np.exp(-((x-len(x)/2)**2+(y-len(x[0])/2)**2)/(2*sigma**2))/(2*np.pi*sigma**2)
        print(gaussian_filter.shape)
        
        #apply the gaussian filter ont the image
        filtered_image = cv2.filter2D(img, -1, gaussian_filter)
        
        #add this new image to the filtered images list
        filtered_images.append(filtered_image)
        
        #below code to visualize the kernel image 
        plt.figure()
        filter_img=plt.imshow(gaussian_filter)
        
        #we can also view the kernel applied image
        cv2.imshow("filtered_image"+str(i), filtered_image/np.max(filtered_image)) #this divide operation with the max pixel length will help us view the image 
        
        # save the kernel image and the blurred image
        kernel_name = "kernel_img_" + str(i+1) + ".jpg"
        blur_image= "blur_img_" + str(i+1) + ".jpg"
        plt.savefig(kernel_name)
        #cv2.imwrite(kernel_name, filter_img)
        cv2.imwrite(blur_image, filtered_image)
        
        #standard wait command
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return filtered_images, sigma_vals
    

def difference_of_gaussian(cutouts):
    """
    the difference of gaussian takes all the 12 images produced in the previous task
    and takes the difference of images meaning that we will now have n-1 images i.e 10 images
    the difference criterian is second image minus the first image
    """
    difference_images=[]
    
    for i,image in enumerate(range(len(cutouts)-1)):
        print(image)
        difference_image=np.subtract(cutouts[image+1],cutouts[image])
        difference_images.append(difference_image)
        
        #show me the difference of gaussian image
        cv2.imshow("difference of gaussian"+str(i), difference_image/np.max(difference_image))
        
        #wait for new key press 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return difference_images
        

def thresholding_func(dog_images,sigmas,threshold=10):
    """
    finds the threshold points in the dog images greater than t=10 [3pts]
    """
    
    key_points=[]
    #traverse through all dog images
    #we are starting from 1 because we also want to look in the previous image 
    for i,j in zip(range(1,len(dog_images)-1),range(1,len(sigmas)-1)):
        #then traverse through all x positions
        for x in range(1,len(dog_images[i])-1):
            #then to all the y positions 
            for y in range(1,len(dog_images[i][0])-1):
                    if ((dog_images[i][x,y]>threshold) and
                    
                        #inside the current image  >
                        (dog_images[i][x,y]>dog_images[i][x-1,y-1]) and
                        (dog_images[i][x,y]>dog_images[i][x-1,y]) and
                        (dog_images[i][x,y]>dog_images[i][x-1,y+1]) and
                        (dog_images[i][x,y]>dog_images[i][x,y-1]) and
                        (dog_images[i][x,y]>dog_images[i][x,y+1]) and
                        (dog_images[i][x,y]>dog_images[i][x+1,y-1]) and
                        (dog_images[i][x,y]>dog_images[i][x+1,y]) and
                        (dog_images[i][x,y]>dog_images[i][x+1,y+1]) and
                        
                        #inside the previous image
                        (dog_images[i][x,y]>dog_images[i-1][x,y]) and
    
                        (dog_images[i][x,y]>dog_images[i-1][x-1,y-1]) and
                        (dog_images[i][x,y]>dog_images[i-1][x-1,y]) and
                        (dog_images[i][x,y]>dog_images[i-1][x-1,y+1]) and
                        (dog_images[i][x,y]>dog_images[i-1][x,y-1]) and
                        (dog_images[i][x,y]>dog_images[i-1][x,y+1]) and
                        (dog_images[i][x,y]>dog_images[i-1][x+1,y-1]) and
                        (dog_images[i][x,y]>dog_images[i-1][x+1,y]) and
                        (dog_images[i][x,y]>dog_images[i-1][x+1,y+1]) and
                        
                        #inside the next image
                        (dog_images[i][x,y]>dog_images[i+1][x,y]) and
          
                        (dog_images[i][x,y]>dog_images[i+1][x-1,y-1]) and
                        (dog_images[i][x,y]>dog_images[i+1][x-1,y]) and
                        (dog_images[i][x,y]>dog_images[i+1][x-1,y+1]) and
                        (dog_images[i][x,y]>dog_images[i+1][x,y-1]) and
                        (dog_images[i][x,y]>dog_images[i+1][x,y+1]) and
                        (dog_images[i][x,y]>dog_images[i+1][x+1,y-1]) and
                        (dog_images[i][x,y]>dog_images[i+1][x+1,y]) and
                        (dog_images[i][x,y]>dog_images[i+1][x+1,y+1])):
                        
                        key_points.append((x,y,sigmas[j]))
                        
    #we can view th keypoints in the for every dog image
    print("this is the length of my all key points",len(key_points))   
    return key_points


def derivative_of_scale_space(cutouts,sigmas):
    """
    this function implements the task F [4 points]

    on all the images of from task subtask b
    
    """
    # part E
    kernel_x = np.array([[1,0,-1]]) # as specified in the asignment
    kernel_y = kernel_x.T
    
    dx_for_all_sigmas={} #this dictionary will contain all the images transformed by kernel_x 
    dy_for_all_sigmas ={} #this dictionary will contain all the images transformed by kernel_y
    
    #apply botht the filters
    for i,(image,sigma) in enumerate(zip(cutouts,sigmas)):
        
        #apply the filter 
        filter_x = cv2.filter2D(image,-1,kernel_x)
        filter_y = cv2.filter2D(image,-1,kernel_y)
        
        dx_for_all_sigmas[sigma] = filter_x
        dy_for_all_sigmas[sigma] = filter_y
        
        cv2.imshow("kernel_x on image "+str(i),filter_x)
        cv2.imshow("kernel_y on image "+str(i),filter_y)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return dx_for_all_sigmas,dy_for_all_sigmas

def gradients_gaussian_weighted(key_points,t):
    """
    1- calculate the gradient length (mqr) and gradient direction (theta) on 7*7 grid on key_points
    across a sampling function defined in the assignment pdf [4pts]
    
    2- gaussian weighing function [3 points]
    
    3- use maximum orientation histograms [1 point]
    
    """
    dx,dy=t[0],t[1]
    orientations = [] #a list containing orientations
    
    for point in key_points:
        
        weighted_grad = []
        thetas = []
        
        for k1 in range(-3,4):   #defines the k parameter
            for k2 in range(-3,4): #defines the k parameter
                
                x = point[0] 
                y = point[1]
                
                #p[2] denotes the image number(1 to 11) hence these are the same sigmas(0,11) for original task a

                sigma = point[2]
                
                #calculate the q and the r using the formula
                q  = int (x+(3/2)*k1*sigma) #make k
                r = int(y+(3/2)*k2*sigma)
                
                #weighted gaussian function
                w_qr = np.exp((-(q**2+r**2)/(9*sigma**2)/2))/((9*np.pi*sigma**2)/2)
                
                # calculate gx and gy
                if (q<len(dx[sigma]) and r<len(dx[sigma][0])):
                    gx = dx[sigma][q,r]
                    gy = dy[sigma][q,r]
                    thetas.append(np.arctan2(gy,gx))
                    mag = np.sqrt(gx**2+gy**2)
                    weighted_grad.append(mag*w_qr)
                    
        # get the maximum values of theta
        orientations.append(thetas[weighted_grad.index(max(weighted_grad))])
        
    print("this is my orientation :",orientations)
    return orientations

def draw_circles_over_keypoints(key_points,resized_image,orientations,state=True):
    """
    1- draws a circle of radius 3 sigma across all the keypoints over image [3 points]
    
    """
    
    if(state):
        #my original colored is image is my resized image
        resized_image = cv2.imread("./Assignment_MV_01_image_1.jpg") 
        
    # points
    for p,order in zip(key_points,orientations):
        
        #we divide by 2 because we want to see our points in the original image and not the resized_img
        if(state==True):
            x = int(p[0]/2)
            y = int(p[1]/2)
        else:
            x = int(p[0])
            y = int(p[1])

        sigma = int(p[2])
        radius=int(3*sigma)
        radian=order
        
        #calculate form single value of orientation
        #radian is orientation value
        #from the previous part
        xBar =  int(np.round(x + radius * np.cos(radian))) # radius with be the lenght of the line
        yBar =  int(np.round(y + radius * np.sin(radian)))
        if(radius>20):
            cv2.circle(resized_image,(y,x),radius,color = (0, 0, 255) ,thickness = 2)
            cv2.line(resized_image,(y,x),(yBar,xBar),color=(0,0,255),thickness=2)
        
    cv2.imshow("circles denoting the key points",resized_image/np.max(resized_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#task 2 functions 

def convert_grey_scale_task_2(image_1,image_2,color=0):
    """
    this function is very similar to resized image function used in task 1 part A
    but this time it takes two images and converts both of them into greyscale images  2pts
    and makes sure the datatype is float32                                             1pts
    it does not do the resizing aspect as in the last task
    """
    #load our images in the grey scale format with only one channel
    image_1= cv2.imread(image_1,color) 
    image_2 = cv2.imread(image_2,color) 
    
    #show the size of the image on the console
    print("="*10)
    print("size of the first image: {}".format(image_1.shape))
    original_height=int(image_1.shape[0])
    original_width=int(image_1.shape[1])
    print("image 1 pixel 1 value : ",image_1[0][0])
    print('Height of Image:', original_height, 'pixels')
    print('Width of Image: ', original_width, 'pixels')
    print("size of the second image: {}".format(image_2.shape))
    original_height=int(image_2.shape[0])
    original_width=int(image_2.shape[1])
    print("image 1 pixel 1 value : ",image_2[0][0])
    print('Height of Image:', original_height, 'pixels')
    print('Width of Image: ', original_width, 'pixels')
    print("="*10)
    
    #convert the image into float32 type
    print("===now converting the image_1 dtype to float===")
    #let's convert this image dtype to float 32 fromunit8
    image_1 = np.float32(image_1)
    print('image dtype: ',image_1.dtype)
    print(image_1[0][0])
    print("===now converting the image_2 dtype to float===")
    #let's convert this image dtype to float 32 fromunit8
    image_2 = np.float32(image_2)
    print('image dtype: ',image_2.dtype)
    print(image_2[0][0])
    
    print("are both images same : ", np.array_equal(image_1,image_2))
    return image_1,image_2

def select_portion_image(image,coordinates,thickness=5):
    """
    this function takes in an image but in normal bgr format
    draws a rectangle across the diagnal coordiantes provided 1pts
    cuts the diagnal image into a new image     2pts
    and finally displays the cut out image 
    """
    image=cv2.imread(image,1) 
    original_image=copy.deepcopy(image)
    
    cv2.imshow("Original Image with boundary", original_image)
    cv2.waitKey(0) 
    #make a rectangle(image,start diagnol ,end diag,color of the line, thickness )

    cv2.rectangle(image, coordinates[0], coordinates[1], (127,50,127), thickness)
    cv2.imshow("Rectangled image", image)
    
    #this will show the bounded boxed image
    cv2.waitKey(0)
    
    #cutting out the image patch
    
    # we now set thickness to 0 as we don't want the highlight section to take any further margin over the cut out
    thickness=0 #
    print("thick: ",thickness)
    
    #when thickness=0 we are only display the window but when give thickness we can also view the boundary
    start_x=coordinates[0][0]-thickness
    start_y=coordinates[0][1]-thickness
    end_y=coordinates[1][1]+thickness
    end_x=coordinates[1][0]+thickness
    print(start_x,start_y,end_x,end_y)
    cropped = original_image[start_y:end_y, start_x:end_x] #y trimming followed by x trinmming 
    #numpy sees them differntly rows become my y and cols become my x
    
    cv2.imshow("Cropped Image with boundary", cropped) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
    cv2.imwrite('temp.png',cropped)
    
    return cropped

def cutout_plot_match_draw(cropped_image,img_2):
    """
    1- calculate the mean and standard deviation of the cut out patch [2 pts]
    
    2- iterate over all patches in the second image and cut out a patch [2 points]
    
    3- find the cross correlation between the two [3pts]
    
    4-select all the potential places where the cross correlation is similar [2 points]
    
    5- select the one with the maximum corelation and draw a rectangle around the patch and display[2 points]
    
    
    """
    
    #plot the histogram of the image

    hist=cv2.calcHist([cropped_image],[0],None,[256],[0,256])
    
    plt.figure("histogram intensity")
    plt.title("pixel intensity of the cutout")
    plt.ylabel("counts of pixels")
    plt.xlabel("pixel intensity 0 being dim 255 means high")
    plt.bar(range(len(hist)),hist.flatten())
    plt.figure("cumsum intensity")
    plt.plot(np.cumsum(hist))
    
    #get the mean and the std of the cutout
    mean=np.mean(cropped_image)
    std=np.std(cropped_image)
    
    print("my mean value of the cut out is : {} and my std dev is : {}".format(mean,std))
    
    #let's cut out all possible patches in the second image 
    
    image_2=cv2.imread(img_2,0) # this means load in grey scale 
    img_rgb=cv2.imread(img_2)  #load this image in colour
    
    #######################
    
    x=0
    y=0
    
    cut_out_height=cropped_image.shape[0]
    cut_out_width=cropped_image.shape[1]
    print("cutout shape",cropped_image.shape)
    
    image_height=image_2.shape[0]
    image_width=image_2.shape[1]
    
    total_patches=[]
    
    i=1
    for y in range(0,image_height-cut_out_height): #verical sliding
        for x in range(0,image_width-cut_out_width): #horizontal sliding
            pt1=(x,y)
            pt2=(x+cut_out_width,y+cut_out_height)
            patch=image_2[pt1[1]:pt2[1],pt1[0]:pt2[0]]
            if(patch.shape!=(90,70)):
                print(patch.shape)
            i+=1
            total_patches.append(patch) 
            
            #we can also view the mean std dev and cross coeff of this patch 
            #this is a time consuming task as it takes too much time to compute these 3 for all patches
            
            #patch_mean=np.mean(patch)
            #patch_std=np.std(patch)
            #result = cv2.matchTemplate(image_2, patch, cv2.TM_CCOEFF)
            #print(result.shape)
            
            #we will not print out for each patch as there are too many patches
            #print(patch_mean,patch_std,result)
            
            #display the patch
            #cv2.imshow("patch"+str(i), patch) 
            #cv2.waitKey(0) 
            #cv2.destroyAllWindows()
    
    print("total patches :{} ".format(i))
    
    last_patch=total_patches[-1]
    
    #some other stuff to make sure every thing is working right (approximately)
    theoritical_lastpatch=image_2[768-cut_out_height:,1024-cut_out_width:]
    print(theoritical_lastpatch.shape)
    print(last_patch.shape)
    cv2.imshow("tlast patch", theoritical_lastpatch) 
    cv2.waitKey(0)
    cv2.imshow("last patch", last_patch) 
    cv2.waitKey(0)
    print(np.array_equal(last_patch,theoritical_lastpatch))
    cv2.destroyAllWindows()
    
    
    #now we did it with the manual matching method let's use open cv's inbuilt function
    #as it is much faster than our implementation of patch matching
    
    #this function actually slides through all the patches in the image automatically and gives
    #us cross correlation coeff for each patch
    
    #but before we begin remember our to be matched templete is a coloured image
    
    ###########>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    #source - https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
    img_rgb = cv2.imread(img_2)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
    
    #manual matching with patches 
    
    def manual_matching(template,total_patches):
        """compares the cropped image with the list of patches"""
        
        mean=np.mean(template)
        std=np.std(template)
        
        match_patch={}
        for patch in total_patches:
            patch_mean=np.mean(patch)
            patch_std=np.std(patch)
            val=np.mean(((template - mean)/std) * ((patch - patch_mean)/patch_std))
            #print("this is my match with the current patch : {}".format(val))
            match_patch[val]=patch
        
        return match_patch
     
    #################### TURN THIS ON TO SEE THE MANUAL METHOD FOR MATCHING ##########
    
    #match_patch=manual_matching(template,total_patches)
    #print("this is my patch len score:{}".format(len(match_patch)))
    
    ##################################################################################
    w, h = template.shape[::-1]
    
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.58
    loc = np.where( res >= threshold)
    #highlights potential matches on the image
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    cv2.imshow("possible matches"+str(len(loc)),img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    #green border to highlight the best match
    
    #let's get the best bounding box where coeff is maximum
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    #Create Bounding Box
    top_left = max_loc #get the top_left diagnal 1 pt
    
    extra_border=50
    bottom_right = (top_left[0] + w, top_left[1] + h) #get the diagnal 2 point value
    cv2.rectangle(img_rgb, top_left, bottom_right, (200,10,20), 2) #plot the bounding box
    
    cv2.putText(img_rgb, 'best match', (bottom_right[0]-extra_border,bottom_right[1]+extra_border), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (200,10,20), 2)
    
    cv2.imshow('best match', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():
    """This is the main function that will run the entire code"""
    print("="*5+"done"+"="*5)
    
    #setting and assigning image locations
    img_1="./Assignment_MV_01_image_1.jpg" #my first image path
    img_2="./Assignment_MV_01_image_2.jpg" #my second image path
    
    
    #this is my taks 1 a
    
    #load the image using the single channel grey command when color =1 it load the color image instead
    resized_image=single_channel_grey(img_1) 
    # note- the resized image is in float32 format hence it need to be converted to uint8 to view with cv2.imshow
    
    #apply task1 b
    cut_outs,sigmas=gaussian_kernels(resized_image,window_size=6)
    
    #task 1 c returns difference of gaussian images
    dog_images=difference_of_gaussian(cut_outs)
    
    
    #task 1 d
    key_points=thresholding_func(dog_images,sigmas)
    
    #task 1 e
    dx,dy=derivative_of_scale_space(cut_outs,sigmas)
    t=(dx,dy)

    #taks 1 f
    orientations=gradients_gaussian_weighted(key_points,t)
    
    #taks 1 g
    draw_circles_over_keypoints(key_points,resized_image,orientations)
    
    #this is my task 2 a
    
    image_1,image_2=convert_grey_scale_task_2(img_1,img_2)
    
    #task 2 b
    #takes my first image which is still float32 and a set of coordinates to draw the circle
    #coodinate 0 indicates the first diagnal point and cordinate 1 indicates the second diagnal point required for the image
    coordinates=((360,210), (430,300)) #x>> means the horizontal axis, y means the vertical axis
    cropped_image=select_portion_image(img_1,coordinates)
    
    #task 2 c
    cutout_plot_match_draw(cropped_image,img_2)
    
    
    print("="*5+"done"+"="*5)
    
#calling the main
main()
