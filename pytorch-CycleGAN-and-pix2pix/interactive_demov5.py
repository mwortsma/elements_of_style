
import numpy as np
import cv2
import torch
from torch.autograd import Variable

import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html

import subprocess

if __name__ == '__main__':

    proc = subprocess.Popen(['python3','interactive_demo_v3_audio.py'],stdout=subprocess.PIPE)


    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    print(opt.dataset_mode)
    #data_loader = CreateDataLoader(opt)
    #dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    #print(opt.results_dir)
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test

    cap = cv2.VideoCapture(0)

    while(True):

        ret, frame = cap.read()

        line = proc.stdout.readline()
        if line != '':
            #the real code does filtering here
            b = line.rstrip()
            b = int(b)
        else:
            b = 0
            break



        img = frame[:,280:1000]
        img  = cv2.resize(img, (256, 256))
        img_swap = np.swapaxes(img, 0,2)
        #print(img.shape)

        row,col,ch= img.shape
        imgcpy = img.copy()
        std = b/10.0
        print(std)
        noise =cv2.randn(imgcpy,(0),std) #+ gauss
        out = img + noise

        t = Variable(torch.from_numpy(img_swap).view(1,3,256,256).float())
        with torch.no_grad():
            '''
            out = model.netG_B(t)

            out = out.view(3,256,256).numpy()

            out = np.swapaxes(out, 0, 2)
            '''
            #out = img
            out  = cv2.resize(out, (720, 720))


            cv2.imshow('frame',out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        '''
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)
        '''

cv2.waitKey(0)
cv2.destroyAllWindows()



cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    line = proc.stdout.readline()
    if line != '':
        #the real code does filtering here
        b = line.rstrip()
        b = int(b)
    else:
        b = 0
        break
    cv2.imwrite('testimg.jpg', frame)
    break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if b > 400:
        cv2.imshow('frame',frame)
    else:
        cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
