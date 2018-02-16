import pygubu

import Tkinter as tk
import pygubu


class Application:
    def __init__(self, master):

        #1: Create a builder
        self.builder = builder = pygubu.Builder()

        #2: Load an ui files
        builder.add_from_file('gui.ui')

        #3: Create the widget using a master as parent
        self.mainwindow = builder.get_object('Frame_1', master)


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    root.mainloop()