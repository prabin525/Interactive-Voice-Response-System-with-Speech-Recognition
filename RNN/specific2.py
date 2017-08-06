"""yo chai home page ma click garexi aauxa...ya chai aba specific functions haru
jastai bank ko lagi matra milne functions haru tala lekhya hunxa 
jasle garda repsonse dina lai sajilo hunxa
tya edit box ko sato chai euta figure rakhne tyo sound ko waveform jasto"""

import sys,os
import subprocess
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import QThread
from time import sleep

from run_rnn import execute

class playThread(QThread):
    def __init__(self):
        QThread.__init__(self)

    def __del__(self):
        self.wait()


###yesma chai maile conditional lyaera rakhde...
    def run(self):
        print("specific-2")
        sleep(0.3)
        print("What is the choice?")
        a = int(execute())
        print('a2: %s' % a)
        print(type(a))
        self.conditional(a)

    def conditional(self,a):
        if (a == 0):
            print("menu is to repeated")
            subprocess.call(['python3','specific-main.py']) #calls a different script
            self.close()


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

        customer = QtGui.QLabel("You are at Customer Support Menu \t" + u'\u0924\u092a\u093e\u0908\u0902 \u0915\u094d\u0938\u094d\u091f\u092e\u0930 \u0938\u092a\u094b\u0930\u094d\u091f \u092e\u0947\u0928\u0941\u092e\u093e \u0939\u0941\u0928\u0941\u0939\u0941\u0928\u094d\u091b \n'
+            '-----------------------------------------------------------------------------------------')
        first = QtGui.QLabel(u'\u0967' + ' Customer Support Representative    ' + u'\u0915\u0938\u094d\u091f\u092e\u0930 \u0938\u092a\u094b\u0930\u094d\u091f \u092a\u094d\u0930\u0924\u093f\u0928\u093f\u0927\u093f')
        second  = QtGui.QLabel(u'\u0968' + ' Manager   ' + u'\u092e\u0947\u0928\u0947\u091c\u0930 ')
        
        back = QtGui.QLabel(u'\u0966' + '   Previous Menu   '+ u'\u092a\u0941\u0930\u093e\u0928\u094b \u092e\u0947\u0928\u0941')                     

        grid.addWidget(customer,1,0)
        grid.addWidget(first,3,0)
        grid.addWidget(second,4,0)
        grid.addWidget(back,7,0)

        self.addButtons(grid)

        self.get_thread = playThread()
        self.get_thread.start()

        self.setWindowTitle('IVR Solutions')
        self.setGeometry(350,100,500,500)
        self.show()

    def addButtons(self,grid):  

        cancelButton = QtGui.QPushButton("Cancel")
        cancelButton.setStyleSheet('QPushButton {color: red;}')
    
        cancelButton.clicked.connect(QtCore.QCoreApplication.instance().quit)

        grid.addWidget(cancelButton,8,0)

    
    
def main():
    app = QtGui.QApplication(sys.argv)
    ex = InitWindow()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()
