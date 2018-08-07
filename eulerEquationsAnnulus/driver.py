import os
import sys

s = " --testCase {}" . format( sys.argv[1] )

###########################################################################

#General parameters that can be adjusted by the user:

s = s + " --clusterStrength 1"

# s = s + " --saveArrays"
s = s + " --saveContours"
# s = s + " --plotFromSaved"

s = s + " --whatToPlot v"
s = s + " --dynamicColorbar"

# s = s + " --phs 5 --pol 4 --stc 9"
# s = s + " --rks 4"
# s = s + " --VL"

s = s + " --nlv {} --dti {}" . format( sys.argv[2], sys.argv[3] )

###########################################################################

#Parameters that are test-case specific and should not be changed by user:

if sys.argv[1] == "bubble" :
    
    s = s + " --topoType trig"
    s = s + " --tf 1000 --saveDel 100"
    s = s + " --hf 3"
    s = s + " --halfWidth np.pi/3e3"
    s = s + " --amp 1000"
    s = s + " --frq 3001"
    s = s + " --steepness 4e-7"
    s = s + " --kx 1 --ky 2"
    s = s + " --innerRadius 6371000 --outerRadius 6381000"
    
elif sys.argv[1] == "densityCurrent" :
    
    s = s + " --topoType GA"
    s = s + " --tf 900 --saveDel 50"
    s = s + " --hf 2"
    s = s + " --halfWidth np.pi/1e3"
    s = s + " --amp 1000"
    s = s + " --frq 3001"
    s = s + " --steepness 1e-7"
    s = s + " --kx 1 --ky 3"
    s = s + " --innerRadius 6371000 --outerRadius 6381000"
    
elif sys.argv[1] == "gravityWaves" :
    
    s = s + " --topoType GA"
    s = s + " --tf 50 --saveDel 5"
    s = s + " --halfWidth np.pi/1e3"
    s = s + " --amp 1000"
    s = s + " --frq 6000"
    s = s + " --steepness 1e-7"
    s = s + " --innerRadius 6371000 --outerRadius 6381000"
    
else :
    
    raise ValueError("Invalid test case string.")

###########################################################################

#Run the code using the input string s:

os.system( "python main.py" + s )

###########################################################################
