# %%
from pathlib import Path

import webbrowser
from urllib.request import urlopen

import time
import threading

import tkinter as tk
import tkinter.ttk as ttk

from PIL import ImageTk, Image

from onstart import CONFIG

# %%
url = 'http://localhost:8693'
# %%

logo_file = Path.cwd().joinpath('./assets/logo.jpg')

log_file = Path.cwd().joinpath('../log/CTDisplay.log')

# %%


class MainGUI():
    def __init__(self, window_title=CONFIG['short_name'], app_name=CONFIG['app_name'], width=500, height=500):
        self.app_name = app_name
        self.root = tk.Tk()

        self.root.title(window_title)
        self.root.geometry('{}x{}'.format(width, height))

        self.layout()

    def _init_frame_logo(self):
        # Make the logo image obj
        self.frame_logo = tk.Frame(self.root, border=True, borderwidth=2)

        # Logo label
        self.label_logo = tk.Label(
            self.frame_logo, text=self.app_name, font=("Arial", 25))
        self.label_logo.pack(side=tk.LEFT)

        # Logo img
        img = Image.open(logo_file.as_posix())
        img = img.resize((50, 50))  # , Image.ANTIALIAS)
        self.logo_img = ImageTk.PhotoImage(img, size=(50, 50))

        # Panel and pack the logo
        panel = tk.Label(self.frame_logo, image=self.logo_img)
        panel.pack(side=tk.RIGHT)

        return self.frame_logo

    def _button_start_app_on_click(self, content=time.ctime()):
        webbrowser.open_new(url)
        content = time.ctime()
        self.text_logger.insert(tk.END, content)

    def _init_frame_controllers(self):
        # Makeup the frame of controllers
        self.frame_controllers = tk.LabelFrame(self.root, text='Controll')

        # Add the buttons
        self.button_start_app = tk.Button(self.frame_controllers,
                                          text="Start App",
                                          state=tk.DISABLED,
                                          command=self._button_start_app_on_click)

        # self.button2 = tk.Button(self.frame_controllers,
        #                          text="Botton 2",
        #                          state=tk.DISABLED)

        # Pack the buttons
        self.button_start_app.pack(side=tk.LEFT, padx=10, pady=3)
        # self.button2.pack(side=tk.LEFT, padx=10, pady=3)

        return self.frame_controllers

    def _init_frame_logger(self):
        # Make the frame of logger
        self.frame_logger = tk.LabelFrame(self.root, text='Information')

        # The bottom label
        self.label_bottom = tk.Label(self.frame_logger,
                                     text='Research Tool. All rights reserved.')
        # Pack the bottom label
        self.label_bottom.pack(side=tk.TOP, fill=tk.X)

        # The content of the logger is a text area with y-scroller
        self.text_logger = tk.Text(
            self.frame_logger, bg='black', fg='lightgreen', height=30)
        self.text_logger_scroll = tk.Scrollbar(
            self.text_logger, command=self.text_logger.yview)
        self.text_logger.configure(yscrollcommand=self.text_logger_scroll.set)

        self.text_logger.image_create("1.0", image=self.logo_img)

        # Pack the text_logger
        self.text_logger.pack(fill=tk.BOTH)
        # Make sure the 'height' setting is valid
        self.text_logger.pack_propagate(0)
        self.text_logger_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        return self.frame_logger

    def layout(self):
        frames = [
            self._init_frame_logo(),
            self._init_frame_controllers(),
            self._init_frame_logger(),
        ]

        for frame in frames:
            separator = ttk.Separator(self.root, orient=tk.HORIZONTAL)
            separator.pack(expand=True, fill=tk.X)
            frame.pack(fill=tk.X, pady=10, padx=5)

    def mainloop(self):
        self._start_updating()
        self.root.mainloop()

    def _update_in_second(self):

        def _str(e, max_length=200):
            s = str(e)
            if len(s) > max_length:
                s = '{}.....{}'.format(s[:max_length-15], s[-10:])
            return s.strip()

        self.resp = None

        _seek = 0
        while True:
            time.sleep(1)

            with open(log_file, 'r') as fp:
                fp.seek(_seek)
                contents = fp.readlines()[-5:]
                _seek = fp.tell()

            self.text_logger.insert(tk.END, '\n'.join(
                [_str(e) for e in contents]))
            self.text_logger.see(tk.END)

            # print('>' * 40)
            # print('>>', self.text_logger.get("1.0", "end-1c"))
            # print('>' * 40)
            # print()

            if self.resp is None:
                try:
                    f = urlopen(url)
                except Exception as err:
                    print(err)
                    continue
                self.resp = f.read()

                self.button_start_app.config(state=tk.ACTIVE)
                print(self.resp[:100])

    def _start_updating(self):
        t = threading.Thread(target=self._update_in_second)
        t.setDaemon(True)
        t.start()


gui = MainGUI()


if __name__ == '__main__':
    gui.mainloop()

# %%
