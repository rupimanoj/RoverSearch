[//]: # (Image References)

### Rock samples detection

Initially image is transformed from RGB color space to hsv colorspace using opencv function `cv2.cvtColor(img,cv2.COLOR_BGR2HSV)`
HSV range for yellow color is explored and found to be in below ranges. <br/>
    hsv_lower = [20,100,100] <br/>
    hsv_upper = [40,255,255] <br/>
Once range of color samples is found, openCV InRange api is used to extract pixels of interest.
Below is the function to extract rock samples from given image.

``` python
def get_sample_mask(img):
    img3_bgr = np.dstack((img[:,:,2],img[:,:,1],img[:,:,0]))
    hsv_image = cv2.cvtColor(img3_bgr,cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([40,255,255])
    mask = cv2.inRange(hsv_image,lower_yellow,upper_yellow)
    mask_img = mask//255
    return mask_img
```

### World map update