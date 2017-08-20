import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

def get_different_masks(img):
	img3_bgr = np.dstack((img[:,:,2],img[:,:,1],img[:,:,0]))
	hsv_image = cv2.cvtColor(img3_bgr,cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([20,100,100])
	upper_yellow = np.array([40,255,255])
	lower_path = np.array([170,170,170])
	upper_path = np.array([255,255,255])
	mask_sample = cv2.inRange(hsv_image,lower_yellow,upper_yellow)
	mask_path =  cv2.inRange(img,lower_path,upper_path)
	not_mask_sample = cv2.bitwise_not(mask_sample) #icludes both path and obstacles
	not_mask_path = cv2.bitwise_not(mask_path) #includes samples and obstacles
	obstacles_mask = cv2.bitwise_and(not_mask_sample,not_mask_path)
	sample_mask = mask_sample//255
	path_mask = mask_path//255
	obstacle_mask = obstacles_mask//255
	return path_mask,sample_mask,obstacle_mask
	
# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
	
	img = Rover.img
	
	# source and destination positions from calibration image
	source = np.float32([[7,143],[310,143],[200,96],[118,96]])
	dst_size = 5
	bottom_offset = 6
	destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
				  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
				  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
				  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
				  ])
	
	# perspective transform to get top view
	warped = perspect_transform(img, source, destination)
	
	#collect rock smaples mask, navigable terrain mask and obstacles mask
	path_mask,sample_mask,obstacle_mask = get_different_masks(warped)
	
	#get pixels wrt image coordinates
	ypos_path, xpos_path = path_mask.nonzero()
	ypos_sample, xpos_sample = sample_mask.nonzero()
	ypos_obstacle,xpos_obstacle = obstacle_mask.nonzero()
	
	#get pixels wrt rover
	x_rover_path,y_rover_path = rover_coords(path_mask)
	x_rover_sample,y_rover_sample = rover_coords(sample_mask)
	x_rover_obstacle, y_rover_obstacle = rover_coords(obstacle_mask)

	#rover position wrt world coordinates
	rover_xpos = Rover.pos[0]
	rover_ypos = Rover.pos[1]
	rover_yaw = Rover.yaw
	
	#get pixels wrt world coordinates
	x_world_path, y_world_path = pix_to_world(x_rover_path, y_rover_path, rover_xpos, rover_ypos, rover_yaw, 200, 10)
	x_world_sample, y_world_sample = pix_to_world(x_rover_sample, y_rover_sample, rover_xpos, rover_ypos, rover_yaw, 200, 10)
	x_world_obstacle, y_world_obstacle = pix_to_world(x_rover_obstacle, y_rover_obstacle, rover_xpos, rover_ypos, rover_yaw, 200, 10)
	
	update = True #update only if pitch and roll are close to zero
	if((Rover.pitch > 0.5 and Rover.pitch < 359.5) or (Rover.roll > 0.5 and Rover.roll < 359.5)):
		update = False	
	
	#update world map only when pitch and roll are close to zero. This confirms rover is in stable condition and improves fidelity
	if(update):
		Rover.vision_image[:,:,:] = 0
		Rover.vision_image[ypos_obstacle,xpos_obstacle,0] = 255 
		Rover.vision_image[ypos_path,xpos_path,2] = 255 
		Rover.vision_image[ypos_sample,xpos_sample,1] = 255
		
		Rover.worldmap[y_world_obstacle,x_world_obstacle,0] += 5
		Rover.worldmap[y_world_path, x_world_path, 2] += 10
		Rover.worldmap[y_world_sample,x_world_sample,1] += 15 
	
	#calculate polar coordinates to get steering angle
	dist , angles = to_polar_coords(x_rover_path, y_rover_path)
	Rover.nav_dists = dist 
	Rover.nav_angles = angles
	return Rover