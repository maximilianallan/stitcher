from stitcher import Stitcher

#l = cv2.imread('../images/image_5.png',0)
#r = cv2.imread('../images/image_6.png',0)
image_list = ["images/image_1.png","images/image_4.png"]

stitcher = Stitcher(image_list)
stitcher.process_images()


matches = matcher.match_keypoints(l_kp,r_kp)
