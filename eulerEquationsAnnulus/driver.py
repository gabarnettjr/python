import os
import sys

s = ""

###########################################################################

#General parameters that can be adjusted by the user:

s = s + " --mlv 1"
s = s + " --tf 1000 --saveDel 100"
s = s + " --clusterStrength 2"

# s = s + " --saveArrays"
s = s + " --saveContours"
# s = s + " --plotFromSaved"

s = s + " --whatToPlot T"
s = s + " --dynamicColorbar"

# s = s + " --phs 5 --pol 4 --stc 9"
# s = s + " --rks 4"
# s = s + " --VL"
s = s + " --nlv 22 --dti 2"

###########################################################################

#Parameters that are test-case specific and should not be changed by user:

if sys.argv[1] == "bubble" :
    
    s = s + " --testCase bubble"
    s = s + " --hf 3"
    s = s + " --halfWidth np.pi/3e3"
    s = s + " --amp 1000"
    s = s + " --frq 3001"
    s = s + " --steepness 4e-7"
    s = s + " --kx 1 --ky 2"
    s = s + " --innerRadius 6371000 --outerRadius 6381000"
    
elif sys.argv[1] == "densityCurrent" :
    
    s = s + " --testCase densityCurrent"
    s = s + " --hf 2"
    s = s + " --halfWidth np.pi/1e3"
    s = s + " --amp 1000"
    s = s + " --frq 3001"
    s = s + " --steepness 1e-7"
    s = s + " --kx 1 --ky 3"
    s = s + " --innerRadius 6371000 --outerRadius 6381000"
    
else :
    
    raise ValueError("Invalid test case string.")

###########################################################################

#Run the code using the input string s:

os.system( "python main.py" + s )

###########################################################################
