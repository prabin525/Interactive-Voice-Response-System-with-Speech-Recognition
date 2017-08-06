import sys,os
import subprocess
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import QThread
from time import sleep

from run_hmm import execute

class playThread(QThread):
    def __init__(self):
        QThread.__init__(self)

    def __del__(self):
        self.wait()


###yesma chai maile conditional lyaera rakhde...
    def run(self):
        print("whats up")
        sleep(0.3)
        print("What is the choice?")
        a = int(execute())
        print('a: %s' % a)
        print(type(a))
        self.conditional(a)

    def conditional(self,a):
        print('conditional')
        if (a == 0):
            print("Repeat menu")         #calls a different script
        if (a == 8):
            print('8 run')
            subprocess.call(['python3','specific1.py'])
        if (a == 9):
            print("not made right now")
            subprocess.call(['python3','specific2.py'])
        if (a == 3):
            print("not made right now")
            subprocess.call(['python3','specific3.py'])

class InitWindow(QtGui.QWidget):
    def __init__(self):
        super(InitWindow, self).__init__()
        self.initUI()

    def initUI(self):

        grid = QtGui.QGridLayout()
        self.setLayout(grid)

        pic = QtGui.QLabel()
        pic.setPixmap(QtGui.QPixmap("logo.jpg"))
        grid.addWidget(pic, 0, 0)

        first = QtGui.QLabel(u'\u0967' + '  Marketing   ' + u'\u092e\u093e\u0930\u094d\u0915\u0947\u091f\u093f\u0919\u200b' )
        second = QtGui.QLabel(u'\u0968' + ' Customer Support    ' + u'\u0915\u0938\u094d\u091f\u092e\u0930 \u0938\u0939\u093e\u092f\u0924\u093e')
        third = QtGui.QLabel(u'\u0969' + '  Technical Support   ' + u'\u091f\u0947\u0915\u094d\u0928\u093f\u0915\u0932 \u0938\u0939\u093e\u092f\u0924\u093e')
        back = QtGui.QLabel(u'\u0966' + '   Repeat Menu   '+ u'\u092e\u0947\u0928\u0941 \u0926\u094b\u0939\u094b\u0930\u094d\u200d\u092f\u093e\u0909\u0928')           

        grid.addWidget(first,1,0)
        grid.addWidget(second,2,0)
        grid.addWidget(third,3,0)
        grid.addWidget(back,5,0)

        self.addButtons(grid)

        # first.mousePressEvent = self.textClicked

        self.get_thread = playThread()
        self.get_thread.start()

        
        self.setWindowTitle('IVR Solutions')
        self.setGeometry(350,100,500,500)

        self.show()



    def addButtons(self,grid):  

        cancelButton = QtGui.QPushButton("Cancel")
        cancelButton.setStyleSheet('QPushButton {color: red;}')
    
        cancelButton.clicked.connect(QtCore.QCoreApplication.instance().quit)

        grid.addWidget(cancelButton,6,0)

    def textClicked(self,event):
        subprocess.call(['python3','specific3.py'])          #calls a different script


def main():
    app = QtGui.QApplication(sys.argv)
    ex = InitWindow()
    # #! /usr/bin/env python
    # from PyQt4.phonon import Phonon
    # # from PyQt4.QtGui import QApplication
    # from PyQt4.QtCore import SIGNAL, SLOT
    # from PyQt4.QtCore import QFile
    # # import sys
    # import signal
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    # QtGui.QApplication.setApplicationName('phonon-play')
    # media = Phonon.MediaObject()
    # audio = Phonon.AudioOutput(Phonon.MusicCategory)
    # Phonon.createPath(media, audio)
    # source = Phonon.MediaSource("sample1.wav")
    # if source.type() != -1:              # -1 stands for invalid file
    #     media.setCurrentSource(source)
    #     # app.connect(media, SIGNAL("finished()"), app, SLOT("quit()"))
    #     media.play()
    #     return app.exec_()
    # else:
    #     return -2
    sys.exit(app.exec_())


if __name__=='__main__':
    main()
