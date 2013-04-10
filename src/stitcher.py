#!/usr/bin/python

"""
A utility for stitching image sequences together

"""
import sys,argparse,cv
from worker import Worker

class Stitcher:

    def __init__(self,im_seq):

        self.ims_ = []

        for f in im_seq:
            try:
                self.ims_.append(cv.LoadImageM(f))
            except IOError as e:
                print e.args                

        if len(self.ims_) < 2:
            raise ValueError("Error, filenames do not correspond to valid images. Need at least 2 image files to stitch.")
        
    def StitchAll(self):
        
        self.cur_im_ = self.ims_[0]

        for image in self.ims_:
            self.StitchSingle(image)


    def StitchSingle(self,image):
        
        worker = Worker()
        
        try:
            worker.FindCorrespondingPoints(self.cur_im_,image)
        except Exception as e:
            print e.args
            exit(1)

        pose = worker.FindRelativePose()

        self.cur_im_ = worker.Remap()
    

        

if __name__ == '__main__':

                 
    im = ['../images/image_1.png','../images/image_2.png']

    try:
        stitcher = Stitcher(im)
        stitcher.StitchAll()
    except ValueError as e:
        print e.args
        exit(1)
#    except Exception as e:
 #       print e.args
  #      exit(1)




    


                                        
                
                                        

                

