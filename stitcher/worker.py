#!/usr/bin/python

class Worker:

    #def __init__(self):


    def FindCorrespondingPoints(self,image_1, image_2):

        self.c_im_ = image_1
        self.n_im_ = image_2
        self.c_point_ = [(1,3),(4,2)]

        if(len(self.c_point_) < 4):
            raise Exception("Unable to find enough corresponding points")


    def FindRelativePose(self,points):

        self.pose_ = [[1,2,3],[2,3,4],[4,5,4]]

    def Remap(self):
        
        return image_1

        
