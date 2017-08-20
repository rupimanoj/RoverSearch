[//]: # (Image References)

### Rock samples detection

Initially image is transformed from RGB color space to hsv colorspace using opencv function `cv2.cvtColor(img,cv2.COLOR_BGR2HSV)` <br/>
HSV range for yellow color is explored and found to be in below ranges. <br/>
    hsv_lower = [20,100,100] <br/>
    hsv_upper = [40,255,255] <br/>
Once range of color samples is found, openCV InRange api `cv2.inRange(hsv_image,lower_yellow,upper_yellow)` is used to extract pixels of interest.
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

### Obstacle detection

Pixels which does not fall under the category of navigable path and rock samples are classified into obstacles.<br/>
Rock samples pixels were found as explained above. <br/>
Navigable terrain samples are found by using default RGB threshold condition of [160,160,160].<br/>
Once navigable terrain and rock samples are known obstacles pixels are found as below.

``` python
#To find navigable terrain
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    color_select = np.zeros_like(img[:,:,0])
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    color_select[above_thresh] = 1
    return color_select
```

``` python

def get_obstacles_mask(path_mask,sample_mask):
    not_mask_sample = cv2.bitwise_not(sample_mask) #icludes both path and obstacles
    not_mask_path = cv2.bitwise_not(path_mask) #includes samples and obstacles
    obstacles_mask = cv2.bitwise_and(not_mask_sample,not_mask_path)
    obstacles_mask = obstacles_mask//255
    return obstacles_mask

```
### World map update