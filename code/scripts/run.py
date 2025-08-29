import sys
<<<<<<< HEAD
sys.path.insert(0, '/code/src') # probably in the local computer we have to do '../src'
print(sys.path) # you compputer search all the folders named package and pick the first one to use the module file, so be careful having duplicated package folders

from package.module import variable # get the info from package module inside src
print(variable)
=======
sys.path.insert(0, '/code/src')

from package.module import jake_variable
print(jake_variable)
>>>>>>> 69019740220f88dff92040140d616f2f5641226d
