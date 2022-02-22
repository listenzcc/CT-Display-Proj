# %%
import os
from pathlib import Path

import webbrowser
from urllib.request import urlopen
import time
import threading
# Import the required libraries
# from tkinter import *
import tkinter as tk
# from PIL import ImageTk, Image

# %%
url = 'http://localhost:8050'
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

        # canvas.pack(fill=tk.BOTH, expand=True)
        canvas.place(x=0, y=0, anchor=tk.NW)

        self.add_label()
        self.add_button()

    def add_label(self):
        self.text = tk.StringVar()
        self.text.set("Test")
        self.label = tk.Label(self.root, textvariable=self.text)
        self.label.place(x=500, y=500, anchor=tk.SE)

    def add_button(self):
        self.button = tk.Button(self.root,
                                text="Start App",
                                command=self.changeText)
        self.button.place(x=10, y=10, anchor=tk.NW)
        self.button.config(state=tk.DISABLED)

        # self.label.pack()

    def changeText(self, content=time.ctime()):
        print(content)
        print(time.ctime())
        webbrowser.open(url)
        content = time.ctime()
        self.text.set(content)

    def mainloop(self):
        self.foo()
        self.root.mainloop()

    def _str(self, e):
        s = str(e)
        max_length = 80
        if len(s) > max_length:
            s = '{}.....{}'.format(s[:max_length-15], s[-10:])
        return s

    def _foo(self):
        self.resp = None

        while True:
            time.sleep(1)
            contents = open(log).readlines()[-5:]
            self.text.set('\n'.join([self._str(e) for e in contents]))

            if self.resp is None:
                try:
                    f = urlopen(url)
                except Exception as err:
                    print(err)
                    continue
                self.resp = f.read()

                self.button.config(state=tk.ACTIVE)
                print(self.resp[:100])

    def foo(self):
        t = threading.Thread(target=self._foo)
        t.setDaemon(True)
        t.start()


app = Test()


def foo():
    os.system("powershell ./serve.ps1")


t = threading.Thread(target=foo)
t.setDaemon(True)
t.start()

app.mainloop()

# %%
