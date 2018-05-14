
import numpy as np
import cv2
import torch
from torch.autograd import Variable

import os
import time
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
import librosa
import librosa.display
import matplotlib.pyplot as plt

if __name__ == '__main__':


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

    cap = cv2.VideoCapture('/Users/mitchellw/Desktop/out.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(cap.isOpened())
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    vout = cv2.VideoWriter('output3.avi',fourcc, fps, (256,256))
    first = True
    frame_num = 0

    y, sr = librosa.load('/Users/mitchellw/Desktop/audio.mp3', duration=120)

    # 3. Run the default beat tracker
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    # 4. Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    idx = 0

    while(True):
        frame_num += 1
        time = frame_num/fps

        ret, frame = cap.read()

        if not ret:
            break

        if time > beat_times[idx]:
            idx += 1

        diff = (beat_times[idx] - time)**3
        print(time, diff)


        img = frame[:,280:1000]
        img_resize  = cv2.resize(img, (256, 256))
        img_resize = img_resize + np.random.normal(0,diff*80, (256, 256, 3))


        img_swap = np.swapaxes(img_resize, 0,2)



        t = Variable(torch.from_numpy(img_swap).view(1,3,256,256).float())
        with torch.no_grad():

            out = model.netG_B(t)


            out = out.view(3,256,256).numpy()

            out = np.swapaxes(out, 0, 2)
            out = (out - out.min())/(out.max() - out.min())

            out = np.uint8(0.4*img_resize + 255*(0.6)*out)
            vout.write(out)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    continue
        prevframe = img.copy()



        '''
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)
        '''

vout.release()
cv2.destroyAllWindows()
