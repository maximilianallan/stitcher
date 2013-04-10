import cv2
import scipy as sp
import sys 

class Matcher:

    def __init__(self):
        
        self.DETECTOR_TYPE = "SURF"
        self.DESCRIPTOR_TYPE = "BRIEF"
        self.MATCHER_TYPE = "BruteForce-Hamming"

    def find_matches(self,left_image,right_image):
        
        left_kpoints = self.find_keypoints(left_image)
        right_kpoints = self.find_keypoints(right_image)

        return (left_kpoints,right_kpoints)

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

    l = cv2.imread('../images/image_5.png',0)
    r = cv2.imread('../images/image_6.png',0)

    if l is None or r is None:
        print "could not open image\n"
        sys.exit(1)

    matcher = Matcher()
    
    (l_kp,r_kp) = matcher.find_matches(l,r)
    matches = matcher.match_keypoints(l_kp,r_kp)

    dist = [m.distance for m in matches]
    thres_dist = (sum(dist) / len(dist)) * 0.5
    sel_matches = [m for m in matches if m.distance < thres_dist]
    
    h1, w1 = l.shape[:2]
    h2, w2 = r.shape[:2]
    view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
    view[:h1, :w1, 0] = l
    view[:h2, w1:, 0] = r
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]
    
    for m in sel_matches:
    # draw the keypoints
        print m.queryIdx, m.trainIdx, m.distance
        color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
        start = (int(l_kp[0][m.queryIdx].pt[0]),int(l_kp[0][m.queryIdx].pt[1]))
        end = (int(r_kp[0][m.trainIdx].pt[0] + w1), int(r_kp[0][m.trainIdx].pt[1]))
#        print start
#        print end
        cv2.line(view, start , end,  color)
        
    cv2.imshow("view", view)
    cv2.waitKey()

