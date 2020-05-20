import os
import time
import random
import numpy as np
import cv2
import tkinter as tk

from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
from inference import build_model, find_foreign_object



def detect():

    path = filedialog.askopenfilename()

    time.sleep(2)

    if os.path.exists(path):

        global panel

        inp_img = cv2.imread(path)
        inp_img = inp_img / 255
        inp_img = np.expand_dims(inp_img, axis=0)

        out_img = model.predict(inp_img)

        inp_img = np.squeeze(inp_img, axis=0)
        out_img = np.squeeze(out_img, axis=0)

        res, is_fod = find_foreign_object(inp_img, out_img, return_is_fod=True)
        
        detected = cv2.resize(res, (500,500), interpolation=cv2.INTER_CUBIC)
        detected = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
        detected = Image.fromarray(detected)
        detected = ImageTk.PhotoImage(detected)

        if panel is None:
            panel = tk.Label(root, image=detected)
            panel.image = detected
            panel.pack(side="bottom")
        else:
            panel.configure(image=detected)
            panel.image = detected

        if is_fod:
            messagebox.showwarning(title="Detekce", message="Pozor, ve scéně se nachází cizí objekt!")

        if not is_fod:
            messagebox.showinfo(title="Detekce", message="Ve scéně se nenachází žádný cizí objekt.")


root = tk.Tk(screenName="Foreign Object Detection")
root.title("Detekce cizích objektů")

detect_button = tk.Button(root, text="Detekovat cizí objekt", font=("Verdana", 10), width=25, bg="SkyBlue1", activebackground="white", command=detect)
detect_button.pack()

panel = None



if __name__ == "__main__":

    model = build_model()

    root.mainloop()