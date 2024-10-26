# Python code for Multiple Color Detection 
  
  
import numpy as np 
import cv2 
  
  
# Capturing video through webcam 
vid = cv2.VideoCapture("traffic_light.mp4")
  
# Start a while loop 
while(1): 
      
    # Reading the video from the 
    # webcam in image frames 
    _, imageFrame = vid.read() 
  
    # Convert the imageFrame in  
    # BGR(RGB color space) to  
    # HSV(hue-saturation-value) 
    # color space 
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 
    kernel = np.ones((5, 5), "uint8") 

    # Set range for red color and  
    # define mask 
    red_lower = np.array([160, 200, 111], np.uint8) 
    red_upper = np.array([180, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
    res_red = cv2.GaussianBlur(red_mask, (5, 5), 2)
    res_red = cv2.dilate(res_red, kernel, iterations=2)
    
    
  
    # Set range for green color and  
    # define mask 
    green_lower = np.array([45, 200, 72], np.uint8) 
    green_upper = np.array([102, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 
    res_green = cv2.GaussianBlur(green_mask, (5, 5), 2)
    res_green = cv2.dilate(res_green, kernel, iterations=2)
  
    # Set range for yellow color and 
    # define mask 
    yellow_lower = np.array([5, 200, 100], np.uint8) 
    yellow_upper = np.array([15, 255, 255], np.uint8) 
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper) 
    res_yellow = cv2.GaussianBlur(yellow_mask, (5, 5), 2)
    res_yellow = cv2.dilate(res_yellow, kernel, iterations=2)
      
    
    colors = ["Red", "Yellow", "Green"]
    res_colors = [res_red, res_yellow, res_green]
    # Creating contour to track color 
    for color, res_color in zip(colors, res_colors):
        
        contours, hierarchy = cv2.findContours(res_color, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 900): 
                x, y, w, h = cv2.boundingRect(contour) 
                    
                cv2.putText(imageFrame, color, (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                            (255, 0, 0), 3)  
          
            
    # Program Termination 
    cv2.imshow("Traffic Light Detection", imageFrame) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cv2.destroyAllWindows() 
        break
