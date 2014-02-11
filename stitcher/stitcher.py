import cv2
import scipy as sp
import sys 
import numpy as np
import math

class MatchedKeypoint:

  def __init__(self,kp1,kp2,distance):
    self.kp1 = kp1
    self.kp2 = kp2
    self.distance = distance


class Stitcher:

    def __init__(self,image_list):
        
        if len(image_list) < 2:
            raise Exception("Error, need to give at least two images.\n")

        self.DETECTOR_TYPE = "SURF"
        self.DESCRIPTOR_TYPE = "BRIEF"
        self.MATCHER_TYPE = "BruteForce-Hamming"

        #process first image pair
        self.image_list_ = image_list

    def process_images(self):

        self.image_list_ = self.load_images(self.image_list_)

        self.patched_image_ = self.image_list_.pop()
                       
        for image in self.image_list_:

            self.patched_image_ = self.stitch_images(self.patched_image_,image)

            
    def sparsify_keypoints(self,keypoints,image_dims):
    
        def inside(pt,tl,br):
          return pt[0] > tl[0] and pt[0] < br[0] and pt[1] > tl[1] and pt[1] < br[1]
        
        step = image_dims[0]/15
       
        sparse_keypoints = []
        
        for r in range(0,image_dims[0],step):
          for c in range(0,image_dims[1],step):
            
            sparse_keypoints_in_region = [ kp for kp in keypoints if inside(kp.kp1, (c,r), (c+step,r+step) ) ]
            sparse_keypoints_in_region = sorted(sparse_keypoints_in_region,key = lambda x: x.distance)
            sparse_keypoints += sparse_keypoints_in_region[0:2]

            
        return sparse_keypoints
        
    def stitch_images(self,image_1,image_2):

        keypoints = (self.find_keypoints(image_1),self.find_keypoints(image_2)) # ( (keypoints1,distances1), (keypoints2,distances2) )
        
        #match the keypoints and then extract the coordiantes and distances 
        matched_keypoints = self.match_keypoints(keypoints[0],keypoints[1])
        matched_keypoints = [ MatchedKeypoint(keypoints[0][0][mkp.queryIdx].pt,keypoints[1][0][mkp.trainIdx].pt,mkp.distance) for mkp in matched_keypoints ]
       
        matched_keypoints = self.sparsify_keypoints(matched_keypoints, image_1.shape)
        matched_keypoints = sorted(matched_keypoints,key = lambda x: x.distance)
        
        good_keypoints = matched_keypoints

        """view_image = np.ndarray(shape=(image_1.shape[0],image_1.shape[1]*2,3),dtype=np.uint8)
        
        for i in range(3):
          view_image[:,0:image_1.shape[1],i] = image_1
          view_image[:,image_2.shape[1]:image_2.shape[1]*2,i] = image_2
        
        for kp in good_keypoints:
          kp2_ = kp.kp2 + np.asarray([image_1.shape[1],0])
          kp2_ = (int(kp2_[0]),int(kp2_[1]))
          kp1_ = (int(kp.kp1[0]),int(kp.kp1[1]))
          cv2.line(view_image,kp1_,kp2_,(244,25,13))
        
        cv2.imshow("view", view_image)
        cv2.waitKey()
        """
        
        transformation = self.find_relative_pose(np.asarray([x.kp1 for x in good_keypoints]),np.asarray([x.kp2 for x in good_keypoints]))
        
        self.remap_pixels(transformation,image_1,image_2)

        

    def find_relative_pose(self,kp1,kp2):
        #homography is for [y,x,1]
        (transformation,mask) = cv2.findHomography(kp1,kp2,cv2.cv.CV_RANSAC)
        
        return transformation
        
        
    def find_bounding_box(self,homography,h1,w1,h2,w2):     
      
      #map the pixels from the corners of the new frame to the coordinate system of the stitched frames
      inverse = np.linalg.inv(homography)
  
      #remap the coordinates of hte image boundaires into the stiched image space
      top_left = np.dot(inverse,np.asarray([0,0,1]))
      top_right = np.dot(inverse,np.asarray([w2,0,1]))
      bottom_left = np.dot(inverse,np.asarray([0,h2,1]))
      bottom_right = np.dot(inverse,np.asarray([w2,h2,1]))
      
      #normalize
      top_left = top_left/top_left[2]
      top_right = top_right/top_right[2]
      bottom_left = bottom_left/bottom_left[2]
      bottom_right = bottom_right/bottom_right[2]
      
      #find the max/min values
      min_x = min(min(top_left[0],bottom_left[0]),0)
      max_x = max(max(top_right[0],bottom_right[0]),w1)
      min_y = min(min(top_left[1],top_right[1]),0)
      max_y = max(max(bottom_left[1],bottom_right[1]),h1)
  
      return (min_x,max_x,min_y,max_y)
      
    def remap_pixels(self, homography, image_1,image_2):
    
        h1, w1 = image_1.shape[:2]
        h2, w2 = image_2.shape[:2]
        
        #project the coordinates of the new image into the space of the initial image
        x_start,x_end,y_start,y_end = self.find_bounding_box(homography,h1,w1,h2,w2)
        
        t_origin = -np.asarray([x_start,y_start])
        
        view = np.zeros((y_end-y_start,x_end-x_start, 3), np.uint8)
        
        
        for r in range(view.shape[0]):
          for c in range(view.shape[1]):
            
            #t_origin + 
            coord = np.asarray([c,r])
            
            try:
              view[r,c] = image_1[coord[1],coord[0]]
            except Exception as e:
              inverse_mapped_pt = np.dot(homography,np.asarray([coord[0],coord[1],1]))
              inverse_mapped_pt = inverse_mapped_pt/inverse_mapped_pt[2]
              try:
                view[r,c] = self.interpolate_from(image_2,inverse_mapped_pt[1],inverse_mapped_pt[0])
              except Exception as e:
                pass


        cv2.imshow("view", view)
        cv2.waitKey()

        
    def interpolate_from(self,image,r,c):
    
        rgb = [0,0,0]
        
        upper_r = math.ceil(r)
        upper_c = math.ceil(c)
        lower_r = math.floor(r)
        lower_c = math.floor(c)
        
        f_r1 = (upper_c - c)*image[lower_r,lower_c] + (c-lower_c)*image[lower_r,upper_c] #interpolate along top row
        f_r2 = (upper_c - c)*image[upper_r,lower_c] + (c-lower_c)*image[upper_r,upper_c] #interpolate along bottom row
        return (upper_r - r)*f_r1 + (r-lower_r)*f_r2 #interpolate along row
        
        
        
    def load_images(self,image_list):

        images = [cv2.imread(image,0) for image in image_list ]
        new_size = (images[0].shape[1]/4,images[0].shape[0]/4)
        images = map(lambda x: cv2.resize(x,new_size),images)

        return images

    def find_keypoints(self,image):

        detector = cv2.FeatureDetector_create(self.DETECTOR_TYPE)
        descriptor = cv2.DescriptorExtractor_create(self.DESCRIPTOR_TYPE)

        kp = detector.detect(image)
        (k,d) = descriptor.compute(image,kp)
        return (k,d)


    def match_keypoints(self, set_1, set_2):

        matcher = cv2.DescriptorMatcher_create(self.MATCHER_TYPE)
        return matcher.match(set_1[1],set_2[1])

    """
        r_threshold = 0.6
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
        flann = cv2.flann_Index(desc2, flann_params)
        idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
        mask = dist[:,0] / dist[:,1] < r_threshold
        idx1 = np.arange(len(desc1))
        pairs = np.int32( zip(idx1, idx2[:,0]) )
        pass
   """ 



    

