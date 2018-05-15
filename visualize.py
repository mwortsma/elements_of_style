import numpy as np
from tkinter import * # do we want python 2.x? if so, use Tkinter
import PIL
from PIL import Image, ImageTk
#import modules.ss_vae as ss_vae

def npToTk(arr):
    ''' convert np array to imageTk format '''
    typedArr = (arr * 255).astype(np.uint8)
    im = Image.fromarray(typedArr)
    im=im.resize((80*int(np.size(arr,1)/28),80),Image.BICUBIC)
    return ImageTk.PhotoImage(im)

class Visualizer:
    def __init__(self, f, window_sz=5, z_sz=50, max_sliders=10):
        '''
        f: single-arg function (only requiring z), outputting an image
        window_sz: how much to each side the sliders can move
        z_sz: length of z vector
        max_sliders: max number of sliders to display
        '''
        self.f = f
        self.max_sliders = max_sliders
        self.z_sz = z_sz
        self.window_sz = window_sz

        self.root = Toplevel() #Tk()
        self.sliders = []

        # will be changed to a new image when sliders change
        npi = f(np.zeros(self.z_sz))
        avgIm = npToTk(npi)
        print('help let me out')
        self.walked=Label(self.root, image=avgIm)
        self.walked.image=avgIm
        self.walked.grid(row=0)

    def refresh(self,event):
        ''' refresh decoded image using values in sliders '''
        z = np.zeros(self.z_sz) # might not have value for all sliders
        for i,slider in enumerate(self.sliders):
            z[i]=slider.get()
        newImg = npToTk(self.f(z))
        self.walked.configure(image=newImg)
        self.walked.image=newImg # prevent garbage collection

    def visualize(self):
        ''' start visualization '''
        # init sliders
        frame=Frame(self.root)
        frame.grid(row=1,column=0, sticky="nsew");
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        for i in range(min(self.z_sz, self.max_sliders)):
            l = Label(frame, text='Dim %d:'%i)
            l.grid(row=i,column=0,sticky='w')
            w = Scale(
                frame, from_=-self.window_sz, to=self.window_sz,
                orient=HORIZONTAL, length=300, command=self.refresh)
            w.set(0)
            w.grid(row=i,column=1,sticky='e')
            self.sliders.append(w)

        self.root.mainloop()
