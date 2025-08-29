import sys
sys.path.insert(0,'/code/src'). #the 0 means ... do this in order, this first (I guess highest priority if have multiple variables or functions named the same thing)
from package.module import variable #that is tricky!  here the .module seems to mean subfolder.  Apparently sometimes use the . to indicate linking to a function within module as well
print(variable)