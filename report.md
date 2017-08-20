[//]: # (Image References)
[perception_step_code]: ./code/perception.py
[decision_step_code]: ./code/decision.py
[training_video]: ./output/test_mapping.mp4
[auto_mode_video]: ./output/rover_autonomous.webm

### Rock samples detection

Initially image is transformed from RGB color space to HSV colorspace using opencv function `cv2.cvtColor(img,cv2.COLOR_BGR2HSV)` <br/>
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

Uusing above mask functions rock samples, navigable terrain and obstacles pixels are obtained. But this pixels positions are with respect to image coordinates.
This coordinates are converted to rover reference frame by using function `rover_coords`. To convert into rover coordinate positions, normal axis flipping and translation techniques are used as explained in tutorial. <br/>

Once we got pixels with respect to rover coordinates, next step is to convert them into world coordinates. To convert into world coordinates, we need to know the position of rover w.r.t world feame. Position and yaw of rover is obtained from data stored in csv file during training. To convert into world coordinates, size of world map is taken as 200 units and scaling factor of 10 is used. <br/>


``` python
#Converting to rover coordinates
x_rover_path,y_rover_path = rover_coords(path_mask)
x_rover_sample,y_rover_sample = rover_coords(sample_mask)
x_rover_obstacle, y_rover_obstacle = rover_coords(obstacle_mask)
```

``` python
# To get rover positions
if data.count < (len(data.xpos) - 1):
	rover_xpos = data.xpos[data.count + 1]
	rover_ypos = data.ypos[data.count + 1]
	rover_yaw = data.yaw[data.count + 1]
else:
	rover_xpos = data.xpos[len(data.xpos) - 1]
	rover_ypos = data.ypos[len(data.ypos) - 1]
	rover_yaw = data.yaw[len(data.yaw) - 1]  
```

``` python
#converting into world coordinates
x_world_path, y_world_path = pix_to_world(x_rover_path, y_rover_path,
                                rover_xpos, rover_ypos, rover_yaw, 200, 10)
x_world_sample, y_world_sample = pix_to_world(x_rover_sample, y_rover_sample,
                                rover_xpos, rover_ypos, rover_yaw, 200, 10)
x_world_obstacle, y_world_obstacle = pix_to_world(x_rover_obstacle, y_rover_obstacle,
                                          rover_xpos, rover_ypos, rover_yaw, 200, 10)											  
```

Once pixel positions are obtained w.r.t. world map, respective channels are updated in `data.worldmap`. Red channel is used for obstacles. Blue is used for navigable terrain. Rock samples are updated with white color.

``` python
data.worldmap[y_world_path, x_world_path, 2] += 5
data.worldmap[y_world_sample,x_world_sample,:] += 10
data.worldmap[y_world_obstacle, x_world_obstacle, 0] += 1
```

<b>Note</b>: With above cumulative addition approach, elements may overflow from 255 and reset to 0.one way to avoid this is instead of cumulative addition set absolute value of 255.
![Training output][training_video]

### Perception step

Perception step is just the combination of all the above code explained. Masks for rock samples, navigable terrain and obstacles were found and world map is updated accordingly. Additionaly we have an vision image, which updates for each frame. Vision image depicts the current image seen by the rover. <br/>

``` python
Rover.vision_image[:,:,:] = 0
Rover.vision_image[ypos_obstacle,xpos_obstacle,0] = 255 
Rover.vision_image[ypos_path,xpos_path,2] = 255 
Rover.vision_image[ypos_sample,xpos_sample,1] = 255
```

One additional thing that is implemented in perception step is world map is updated only when the roll and pitch of rover are close to zero. This improves the fidelity of the mapped region. <br/>

``` python
update = True #update only if pitch and roll are close to zero
if((Rover.pitch > 0.5 and Rover.pitch < 359.5) or (Rover.roll > 0.5 and Rover.roll < 359.5)):
	update = False
```

Navigable terrain pixels w.r.t rover coordinates are converted into polar coordinates and updated in Rover state variable. Obtained polar angles are used in decision step to determine the steering angle. <br/>

``` python
dist , angles = to_polar_coords(x_rover_path, y_rover_path)
Rover.nav_dists = dist 
Rover.nav_angles = angles
```

![perception step code][perception_step_code]

### Decision step

There were not many changes required on decision step function. Default implementation given as part of project artifacts would be suffice. <br/>
By default steering angles were capped to -15 on lower exterme and 15 to higher exterme. This is changed to -20 and 20. <br/>
One more deviation from default implementation is made to make rover follow along the wall. For steering angle, an offset of +10 is added after obtaining mean steering angle. This make rover to follow along the left wall and increases the probability of finding rock samples.

![decision step code][decision_step_code]

### Drive rover

In Autonomous mode, after many trial runs on average rover is able to map about 45 percentage of map with fidelity around 85 percentage. All the rock samples that were part of mapped area were getting detected.

![Autonomous video][auto_mode_video]

### Incomplete Work and Improvements

* Explore unmapped areas. Right now rover is unable to map entire area even with wall following technique. Rover is continuously exploring already mapped areas.
* Pick rock samples. Even though rock samples are getting detected, these is no code implemented to pick rocks.
* If rover goes into stuck state, no recovery techniques were used.