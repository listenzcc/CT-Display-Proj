# %%
import os
from pathlib import Path

import webbrowser

import time
import threading
# Import the required libraries
# from tkinter import *
import tkinter as tk
# from PIL import ImageTk, Image

# %%

folder = Path().joinpath(os.environ['ONEDRIVE'], 'Pictures', 'DesktopPictures')
files = [e for e in folder.iterdir() if e.is_file()]
png = [e for e in files if e.as_posix().endswith('.png')][0]

log = Path(__file__).joinpath('../../log/CTDisplay.log')

# %%


class Test():
    def __init__(self, title='No Title', width=500, height=500):
        self.lst = []

        self.root = tk.Tk()

        self.root.title(title)
        self.root.geometry('{}x{}'.format(width, height))

        img = tk.PhotoImage(file=png, width=width, height=height)

        canvas = tk.Canvas(
            self.root,
            width=width,
            height=width
        )

        canvas.create_image(
            0,
            0,
            anchor=tk.NW,
            image=img
        )

        canvas.pack(fill=tk.BOTH, expand=True)
        # canvas.place(x=0, y=0, anchor=tk.NW)

        self.add_label()
        self.add_button()

    def add_label(self):
        self.text = tk.StringVar()
        self.text.set("Test")
        self.label = tk.Label(self.root, textvariable=self.text)
        self.label.place(x=500, y=500, anchor=tk.SE)

    def add_button(self):
        self.button = tk.Button(self.root,
                                text="Click to change text below",
                                command=self.changeText)
        self.button.place(x=10, y=10, anchor=tk.NW)

        # self.label.pack()

    def changeText(self, content=time.ctime()):
        print(content)
        print(time.ctime())
        webbrowser.open('http://localhost:8050')
        content = time.ctime()
        self.text.set(content)

    def mainloop(self):
        self.foo()
        self.root.mainloop()

    def _foo(self):
        while True:
            time.sleep(1)
            contents = open(log).readlines()[-5:]
            self.text.set('\n'.join([str(e) for e in contents]))

    def foo(self):
        t = threading.Thread(target=self._foo)
        t.setDaemon(True)
        t.start()


app = Test()

app.mainloop()

# %%
# folder = Path().joinpath(os.environ['ONEDRIVE'], 'Pictures', 'DesktopPictures')
# files = [e for e in folder.iterdir() if e.is_file()]
# png = [e for e in files if e.as_posix().endswith('.png')][0]

# %%
# root = Tk()
# root.title('PythonGuides')
# root.geometry('500x500')


# def get_value():
#     name = Text_Area.get()
#     # creating a new window
#     root2 = Tk()
#     root2.geometry("500x500")
#     root2.title("Include Help")
#     # setting the Label in the window
#     label2 = Label(root2, text=f"Welcome To Include Help {name}")
#     label2.place(x=160, y=80)
#     root2.mainloop()


# canvas = Canvas(
#     root,
#     width=500,
#     height=500
# )

# canvas.pack(fill=BOTH, expand=True)

# img = PhotoImage(file=png, width=500, height=500)

# canvas.create_image(
#     0,
#     0,
#     anchor=NW,
#     image=img
# )

# # set the string variable
# Text_Area = StringVar()

# # create a label
# label = Label(root, text="Dynamic Label")

# # placing the label at the right position
# label.place(x=10, y=10, anchor=NW)

# # creating the text area
# # we will set the text variable in this
# Input = Entry(root, textvariable=Text_Area, width=30)
# Input.place(x=130, y=100)

# # create a button
# button = Button(root, text="Submit", command=get_value, bg="green")
# button.place(x=180, y=130)

# root.mainloop()


# # %%
