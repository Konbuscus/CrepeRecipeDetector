import os, sys, inspect
sys.path.append('../Core')
import core

import Tkinter as tk
from tkinter import filedialog
import pygubu


class Application:
    def __init__(self, master):
        #1: Create a builder
        self.builder = builder = pygubu.Builder()

        #2: Load an ui files
        builder.add_from_file('gui.ui')

        #3: Create the widget using a master as parent
        self.mainwindow = builder.get_object('Frame_1', master)

        builder.connect_callbacks(self)

        self.builder = builder

    def on_file_choose_click(self):
        file_path = filedialog.askopenfilename()

        tb = self.builder.get_object('Path')
        tb.delete(0, 1000)
        tb.insert(0, file_path)

    def on_add_click(self):
        tb = self.builder.get_object('Path')
        path = tb.get()
        if path == '':
            return
        try:
            print 'ok !'

            #print self.builder.objects
            fields = []
            for i in self.builder.objects:
                if i == 'Path':
                    continue
                obj = self.builder.objects[i]
                if type(obj) == pygubu.builder.tkstdwidgets.TKEntry:
                    tmp = self.builder.get_object(i)
                    fields.append([tmp.get(), int(tmp.grid_info()['row'])])
                    tmp.delete(0, 1000)
                    tmp.insert(0, '0')
            fields = sorted(fields, key=lambda x:x[1])
            fields = [x[0] for x in fields]

            f = open(path, 'a')
            f.write('\n' + ';'.join(fields))
            f.close()
        except IOError:
            print 'error'

    def on_load_CSV_click(self):
        tb = self.builder.get_object('Path')
        path = tb.get()

        try:
            core.init(path)
            print 'Initialised !'

            print 'Base weights : '
            print core.hiddenLayerWeights

            for epoch in range(50):
                for i in range(len(core.inputData)):
                    data = core.inputData[i]

                    #print 'Load line ', i
                    tmpFinalPrediction, tmpFinalError, tmpFinalX, tmpPredictions, tmpErrors, tmpFinalXs = core.learnOne(data)
                    print "prediction : ", tmpFinalPrediction, " | true : ", data[-1]
                    #print "error : ", tmpFinalError

                    core.retropropagation(tmpErrors, tmpFinalError, data)

            #print 'Final weights : '
            #print core.hiddenLayerWeights

        except IOError:
            print 'error'

    def on_check_recipe_click(self):
        fields = []
        for i in self.builder.objects:
            if i == 'Path':
                continue
            obj = self.builder.objects[i]
            if type(obj) == pygubu.builder.tkstdwidgets.TKEntry:
                tmp = self.builder.get_object(i)
                fields.append([tmp.get(), int(tmp.grid_info()['row'])])
        fields = sorted(fields, key=lambda x: x[1])
        data = [float(x[0]) for x in fields]
        for i in range(len(data)):
            data[i] /= core.inputsDividers[i]

        tmpFinalPrediction, tmpFinalError, tmpFinalX, tmpPredictions, tmpErrors, tmpFinalX = core.learnOne(data)

        print "Predict : ", tmpFinalPrediction

if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    root.mainloop()