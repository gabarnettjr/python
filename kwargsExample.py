#!/usr/bin/python3

###########################################################################

#Example of how you can use kwargs:

#This function takes in either
#(1) a slope and a y-intercept, or
#(2) the x and y coordinates of two points (x0,y0) and (x1,y1),

#and it gives back either
#(1) the equation of the line in slope-intercept form, or
#(2) the equation of the line in point-slope form.

import sys

def getEquationOfLine( formatString="2.5f", **kwargs ) :
    #Parse input:
    for k in kwargs.keys() :
        if k == "slope" :
            m = kwargs[k]
        elif k == "yIntercept" :
            b = kwargs[k]
        elif k == "x0" :
            x0 = kwargs[k]
        elif k == "x1" :
            x1 = kwargs[k]
        elif k == "y0" :
            y0 = kwargs[k]
        elif k == "y1" :
            y1 = kwargs[k]
        elif k == "p0" :
            x0 = kwargs[k][0]
            y0 = kwargs[k][1]
        elif k == "p1" :
            x1 = kwargs[k][0]
            y1 = kwargs[k][1]
        else:
            sys.exit("\nUnexpected keyword argument.\n")
    #Determine if m and b are defined, or if x0,x1,y0,y1 are defined, and
    #then print the equation of the line in the appropriate form (either
    #slope-intercept or point-slope):
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
            % ( formatString, formatString, formatString )
            # print( tmp . format(y1,m,x1) )
            equationString = tmp.format(y1,m,x1)
    else:
        tmp = "y = {0:%s}*x + {1:%s}" % ( formatString, formatString )
        # print( tmp . format(m,b) )
        equationString = tmp.format(m,b)
    return equationString
    
fs = "1.2f"
eq1 = getEquationOfLine( fs, slope=2, yIntercept=1 )
eq2 = getEquationOfLine( fs, p0=(0,1), p1=(1,3) )
eq3 = getEquationOfLine( fs, x0=0, y0=1, p1=(1,3) )
print(eq1)
print(eq2)
print(eq3)

###########################################################################
