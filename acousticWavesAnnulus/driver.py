#!usr/bin/python3
import os

tf = "20.0"
saveDel = "2"

for i in range( 5 ) :
    
    nlv = '{0:1d}'.format( 12*2**i + 2 )
    dti = '{0:1d}'.format( 1*2**i )
    
    #################################
    
    #p5:
    
    st = "--tf " + tf + " --saveDel " + saveDel \
    + " --phs 7 --pol 5 --stc 13 --ptb 35 --rks 3" \
    + " --nlv " + nlv + " --dti " + dti + " --saveArrays"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = "--tf " + tf + " --saveDel " + saveDel \
    + " --phs 7 --pol 5 --stc 13 --ptb 35 --rks 4" \
    + " --nlv " + nlv + " --dti " + dti + " --saveArrays"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = "--tf " + tf + " --saveDel " + saveDel \
    + " --phs 7 --pol 5 --stc 13 --ptb  0 --rks 3" \
    + " --nlv " + nlv + " --dti " + dti + " --saveArrays"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = "--tf " + tf + " --saveDel " + saveDel \
    + " --phs 7 --pol 5 --stc 13 --ptb  0 --rks 4" \
    + " --nlv " + nlv + " --dti " + dti + " --saveArrays"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    #################################
    
    #p3:
    
    st = "--tf " + tf + " --saveDel " + saveDel \
    + " --phs 5 --pol 3 --stc 7 --ptb 35 --rks 3" \
    + " --nlv " + nlv + " --dti " + dti + " --saveArrays"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = "--tf " + tf + " --saveDel " + saveDel \
    + " --phs 5 --pol 3 --stc 7 --ptb 35 --rks 4" \
    + " --nlv " + nlv + " --dti " + dti + " --saveArrays"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = "--tf " + tf + " --saveDel " + saveDel \
    + " --phs 5 --pol 3 --stc 7 --ptb  0 --rks 3" \
    + " --nlv " + nlv + " --dti " + dti + " --saveArrays"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()
    
    st = "--tf " + tf + " --saveDel " + saveDel \
    + " --phs 5 --pol 3 --stc 7 --ptb  0 --rks 4" \
    + " --nlv " + nlv + " --dti " + dti + " --saveArrays"
    print( st + " :" )
    os.system( "python main.py " + st )
    print()

###########################################################################

#reference solution:

i = 5
nlv = '{0:1d}'.format( 12*2**i + 2 )
dti = '{0:1d}'.format( 1*2**i )

st = "--tf " + tf + " --saveDel " + saveDel \
+ " --phs 7 --pol 5 --stc 13 --ptb 0 --rks 4" \
+ " --nlv " + nlv + " --dti " + dti + " --saveArrays"
print( st + " :" )
os.system( "python main.py " + st )
print()

###########################################################################
