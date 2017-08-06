"""yo chai home page ma click garexi aauxa...ya chai aba specific functions haru
jastai bank ko lagi matra milne functions haru tala lekhya hunxa 
jasle garda repsonse dina lai sajilo hunxa
tya edit box ko sato chai euta figure rakhne tyo sound ko waveform jasto"""

import sys,os
import subprocess
from PyQt4 import QtGui, QtCore

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

        voice = QtGui.QLabel("You are at Voice Message Menu \t" + u'\u0924\u092a\u093e\u0908\u0902 \u092d\u094b\u0908\u0938 \u092e\u0947\u0938\u0947\u091c \u092e\u0947\u0928\u0941\u092e\u093e \u0939\u0941\u0928\u0941\u0939\u0941\u0928\u094d\u091b \n'
 +
            '-----------------------------------------------------------------------------------------')
        first = QtGui.QLabel('Please Leave Your Voice Message \t ' + u'\u0906\u092b\u094d\u0928\u094b \u092d\u094b\u0908\u0938 \u092e\u0947\u0938\u0947\u091c \u092c\u094b\u0932\u094d\u0928\u0941\u0939\u094b\u0938 ')
        
        grid.addWidget(voice,1,0)
        grid.addWidget(first,3,0)
       
        self.addButtons(grid)

        self.setWindowTitle('IVR Solutions')
        self.setGeometry(350,100,500,500)
        self.show()

    def addButtons(self,grid):  

        cancelButton = QtGui.QPushButton("End")
        cancelButton.setStyleSheet('QPushButton {color: red;}')
    
        cancelButton.clicked.connect(QtCore.QCoreApplication.instance().quit)
        grid.addWidget(cancelButton,8,0)

    
def main():
    app = QtGui.QApplication(sys.argv)
    ex = InitWindow()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()