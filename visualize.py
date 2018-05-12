import numpy as np
from Tkinter import * # do we want python 2.x? if so, use Tkinter
 import Image, ImageTk
#import modules.ss_vae as ss_vae

max_args = 10

class Visualizer:
    __init__(self, f, x, mu, sigma):
        '''
        f: single-arg function (only requiring z), outputting an image
        x: original image, as np array
        mu, sigma: specifications for distribution to wander around (np arrays)
        '''
        self.f = f
        self.x = x
        self.mu = mu
        self.sigma = sigma

        self.root = Tk()
        self.sliders = []

        # out.data.numpy.()
        # original image in upper left
        img=Image.fromarray(x)
        imgTk=ImageTk.PhotoImage(img)
        original=Label(self.root, image=imgTk)
        original.image=imgTk
        original.grid(row=0,column=0)

        # will be changed to a new image when sliders change
        avg = Image.fromarray(f(self.mu)) # sample for when z=mu
        avgTk = ImageTk.PhotoImage(avg)
        self.walked=Label(self.root, image=avgTk)
        self.walked.image=avgTk
        self.walked.grid(row=0,column=1)

    def refresh():
        ''' refresh decoded image using values in sliders '''
        z = np.array(self.mu) # might not have value for all sliders, bc limited space
        for i,slider in self.sliders:
            z[i]=slider.get() # XXX double check that this makes sense
        img = f(z)
        self.walked.configure(image=img)
        self.walked.image=img # prevent garbage collection

    def visualize():
        ''' start visualization '''
        # init sliders
        frame=Frame(self.root)
        frame.grid(row=1,column=0, sticky="nsew");
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        for i in range(min(len(mu), max_args)): # XXX limited number of sliders-- fix this
            l = Label(frame, text='Dim %s:'%i)
            l.grid(row=i,column=0,sticky='w')
            w = Scale(
                frame, from_=mu[i]-sigma[i], to=mu[i]+sigma[i],
                orient=HORIZONTAL, length=300, command=refresh)
            w.set(mu[i])
            w.grid(row=i,column=1,sticky='e')
            self.sliders.append(w)

        self.root.mainloop()
