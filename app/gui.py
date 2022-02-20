# %%
import os
from pathlib import Path

# Import the required libraries
from tkinter import *
# from PIL import ImageTk, Image

# %%
folder = Path().joinpath(os.environ['ONEDRIVE'], 'Pictures', 'DesktopPictures')
files = [e for e in folder.iterdir() if e.is_file()]
png = [e for e in files if e.as_posix().endswith('.png')][0]

# %%
ws = Tk()
ws.title('PythonGuides')
ws.geometry('500x500')

canvas = Canvas(
    ws,
    width=500,
    height=500
)

canvas.pack(fill=BOTH, expand=True)

img = PhotoImage(file=png, width=500, height=500)

canvas.create_image(
    10,
    10,
    anchor=NW,
    image=img
)

ws.mainloop()


# %%
# # Create an instance of Tkinter Frame
# win = Tk()

# # Set the geometry of Tkinter Frame
# win.geometry("700x450")

# # Open the Image File
# bg = ImageTk.PhotoImage(file=png)

# # Create a Canvas
# canvas = Canvas(win, width=700, height=3500)
# canvas.pack(fill=BOTH, expand=True)

# # Add Image inside the Canvas
# canvas.create_image(0, 0, image=bg, anchor='nw')

# # Function to resize the window


# def resize_image(e):
#     global image, resized, image2
#     # open image to resize it
#     image = Image.open(png)
#     # resize the image with width and height of root
#     resized = image.resize((e.width, e.height), Image.ANTIALIAS)

#     image2 = ImageTk.PhotoImage(resized)
#     canvas.create_image(0, 0, image=image2, anchor='nw')


# # Bind the function to configure the parent window
# win.bind("<Configure>", resize_image)
# win.mainloop()

# %%
