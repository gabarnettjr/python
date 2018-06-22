#!/usr/bin/python3

###########################################################################

#Example of how you can use kwargs:

#This function takes in either:
#(1) a slope and a y-intercept, or
#(2) the x and y coordinates of two points (x0,y0) and (x1,y1)

import sys

def printEquationOfLine( formatString="2.5f", **kwargs ) :
    #Parse input:
    for x in kwargs.keys() :
        if x == "slope" :
            m = kwargs[x]
        elif x == "yIntercept" :
            b = kwargs[x]
        elif x == "x0" :
            x0 = kwargs[x]
        elif x == "x1" :
            x1 = kwargs[x]
        elif x == "y0" :
            y0 = kwargs[x]
        elif x == "y1" :
            y1 = kwargs[x]
        else :
            sys.exit("\nUnexpected keyword argument.\n")
    #Determine if m and b are defined, or if x0,x1,y0,y1 are defined, and
    #then print the line in the appropriate form (either slope-intercept
    #or point-slope):
    try:
        m, b
    except:
        try:
            x0, x1, y0, y1
        except:
            sys.exit("\nBad combination of variables given.\n")
        else:
            m = ( y1 - y0 ) / ( x1 - x0 )
            tmp = "y - {0:%s} = {1:%s} * ( x - {2:%s} )" \
            % (formatString,formatString,formatString)
            print( tmp . format(y1,m,x1) )
    else:
        tmp = "y = {0:%s}x + {1:%s}" % (formatString,formatString)
        print( tmp . format(m,b) )
    
fs = "1.2f"
printEquationOfLine( fs, slope=2, yIntercept=1 )
printEquationOfLine( fs, yIntercept=5, slope=10 )
printEquationOfLine( fs, x0=0, y0=1, x1=1, y1=3 )

###########################################################################
