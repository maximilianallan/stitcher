import cv2
import scipy as sp
import sys 
import numpy as np

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


    def stitch_images(self,image_1,image_2):

        keypoints = (self.find_keypoints(image_1),self.find_keypoints(image_2))
        matched_keypoints = self.match_keypoints(keypoints[0],keypoints[1])

        matched_keypoints = sorted(matched_keypoints,key = lambda x: x.distance, reverse=True)

        good_keypoints = [ ((keypoints[0][0][mkp.queryIdx].pt),(keypoints[1][0][mkp.trainIdx].pt)) for mkp in matched_keypoints[:10] ]

        #print len(keypoints[0][0])
        #print len(keypoints[1][0])

        #for mkp in matched_keypoints[:10]:
#            try:
#                pt1 = (keypoints[0][mkp.queryIdx].pt)
#                pt2 = (keypoints[1][mkp.trainIdx].pt)
 #           except:
 #               print mkp.queryIdx
  #              print mkp.trainIdx
   #         good_keypoints.append( pt1,pt2 )

        self.find_relative_pose(np.asarray([x[0] for x in good_keypoints]),np.asarray([x[1] for x in good_keypoints]))
        
        #tt = []
        #for mkp in matched_keypoints[:10]:
        #    tt.append(


        #for matched_keypoints

        

    def find_relative_pose(self,kp1,kp2):

        (transformation,mask) = cv2.findHomography(kp1,kp2)
        
        x = np.ndarray((3,1))
        x[0,0] = kp2[0,0]
        x[1,0] = kp2[0,1]
        x[2,0] = 1

        #x = np.asarray([kp1[0,0],kp1[0,  1],1])
        print x
        print transformation.dot(x)
        print kp[0,:]

        
        sys.exit(1)

        thres_dist = (sum(dist) / len(dist)) * 0.5
        sel_matches = [m for m in matched_keypoints if m.distance < thres_dist]
    
        h1, w1 = image_1.shape[:2]
        h2, w2 = image_2.shape[:2]
        view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
        view[:h1, :w1, 0] = image_1
        view[:h2, w1:, 0] = image_2
        view[:, :, 1] = view[:, :, 0]
        view[:, :, 2] = view[:, :, 0]
    
        l_kp = keypoints[0]
        r_kp = keypoints[1]

        for m in sel_matches:
        # draw the keypoints
            color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
            start = (int(l_kp[0][m.queryIdx].pt[0]),int(l_kp[0][m.queryIdx].pt[1]))
            end = (int(r_kp[0][m.trainIdx].pt[0] + w1), int(r_kp[0][m.trainIdx].pt[1]))
    #        print start
    #        print end
            cv2.line(view, start , end,  color)
        
        cv2.imshow("view", view)
        cv2.waitKey()

    def load_images(self,image_list):

        images = [cv2.imread(image,0) for image in image_list ]
        new_size = (images[0].shape[0]/4,images[0].shape[1]/4)
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




if __name__ == '__main__':

    #l = cv2.imread('../images/image_5.png',0)
    #r = cv2.imread('../images/image_6.png',0)
    image_list = ["../images/image_5.png","../images/image_6.png"]

    stitcher = Stitcher(image_list)
    stitcher.process_images()


    matches = matcher.match_keypoints(l_kp,r_kp)

    

