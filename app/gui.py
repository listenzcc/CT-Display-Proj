# %%
import os
from pathlib import Path

import webbrowser
from urllib.request import urlopen
import time
import threading
import tkinter as tk
from PIL import ImageTk, Image

# %%
url = 'http://localhost:8050'
# %%

logo_file = Path.cwd().joinpath('./assets/logo.png')

log_file = Path.cwd().joinpath('../log/CTDisplay.log')

# %%


class Test():
    def __init__(self, title='No Title', width=500, height=500):
        self.lst = []

        self.root = tk.Tk()

        self.root.title(title)
        self.root.geometry('{}x{}'.format(width, height))
        self.root.configure(background='grey')

        self.frame1 = tk.Frame(self.root)

        img = Image.open(logo_file.as_posix())
        img = img.resize((50, 50))  # , Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(img, size=(50, 50))
        panel = tk.Label(self.frame1, image=self.img)

        self.add_button()
        self.add_text()

        self.frame1.pack(fill=tk.X)

        self.button.pack(side=tk.LEFT, padx=10, pady=3)
        self.button2.pack(side=tk.LEFT, padx=10, pady=3)

        self.text.pack(fill=tk.BOTH)
        self.text.pack_propagate(0)

        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)

        panel.pack(side=tk.RIGHT)

    def add_text(self):
        self.text = tk.Text(self.root, bg='black', fg='lightgreen', height=30)
        self.scroll = tk.Scrollbar(self.text, command=self.text.yview)
        self.text.configure(yscrollcommand=self.scroll.set)
        return

    def add_button(self):
        self.button = tk.Button(self.frame1,
                                text="Start App",
                                state=tk.DISABLED,
                                command=self.update_log)

        self.button2 = tk.Button(self.frame1,
                                 text="Botton 2",
                                 state=tk.DISABLED)
        return

    def update_log(self, content=time.ctime()):
        webbrowser.open(url)
        content = time.ctime()
        self.text.insert(tk.END, content)

    def mainloop(self):
        self.foo()
        self.root.mainloop()

    def _str(self, e):
        s = str(e)
        max_length = 200
        if len(s) > max_length:
            s = '{}.....{}'.format(s[:max_length-15], s[-10:])

        return s.strip()

    def _foo(self):
        self.resp = None

        while True:
            time.sleep(1)
            contents = open(log_file).readlines()[-5:]
            self.text.insert(tk.END, '\n'.join(
                [self._str(e) for e in contents]))

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


if __name__ == '__main__':

    def foo():
        os.system("powershell ./serve.ps1")

    # t = threading.Thread(target=foo)
    # t.setDaemon(True)
    # t.start()

    app.mainloop()

# %%
