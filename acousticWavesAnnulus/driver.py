#!usr/bin/python3
import os

tf = "20.0"
saveDel = "2"

for i in range( 5 ) :
    
    nlv = '{0:1d}'.format( 12*2**i + 2 )
    dti = '{0:1d}'.format( 1*2**i )
    
    tmp = "--tf " + tf + " --saveDel " + saveDel \
    + " --nlv " + nlv + " --dti " + dti + " --saveArrays"
    
    #################################
    
    #p5:
    
    st = tmp + " --phs 7 --pol 5 --stc 13 --pta 35 --ptr 35 --rks 3"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = tmp + " --phs 7 --pol 5 --stc 13 --pta 35 --ptr 35 --rks 4"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = tmp + " --phs 7 --pol 5 --stc 13 --pta  0 --ptr  0 --rks 3"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = tmp + " --phs 7 --pol 5 --stc 13 --pta  0 --ptr  0 --rks 4"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    #################################
    
    #p4:
    
    st = tmp + " --phs 5 --pol 4 --stc 9 --pta 35 --ptr 35 --rks 3"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = tmp + " --phs 5 --pol 4 --stc 9 --pta 35 --ptr 35 --rks 4"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = tmp + " --phs 5 --pol 4 --stc 9 --pta  0 --ptr  0 --rks 3"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = tmp + " --phs 5 --pol 4 --stc 9 --pta  0 --ptr  0 --rks 4"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    #################################
    
    #p3:
    
    st = tmp + " --phs 5 --pol 3 --stc 7 --pta 35 --ptr 35 --rks 3"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = tmp + " --phs 5 --pol 3 --stc 7 --pta 35 --ptr 35 --rks 4"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = tmp + " --phs 5 --pol 3 --stc 7 --pta  0 --ptr  0 --rks 3"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = tmp + " --phs 5 --pol 3 --stc 7 --pta  0 --ptr  0 --rks 4"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()

###########################################################################

#reference solution:

i = 5
nlv = '{0:1d}'.format( 12*2**i + 2 )
dti = '{0:1d}'.format( 1*2**i )

st = "--tf " + tf + " --saveDel " + saveDel   \
+ " --phs 7 --pol 5 --stc 13 --pta 0 --ptr 0 --rks 4" \
+ " --nlv " + nlv + " --dti " + dti + " --saveArrays"
print( st + " :" )
os.system( "python main.py " + st )
print()

###########################################################################
