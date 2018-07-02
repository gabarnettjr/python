#!/usr/bin/python3
import argparse
import numpy as np

#initialize parser:
parser = argparse.ArgumentParser()

###########################################################################

#add positional argument called "stringToPrint":
parser.add_argument("stringToPrint" \
, help="print the string you put here")

#add positional argument called "numToSquare":
parser.add_argument("numToSquare", type=int \
, help="print the square of the number you put here")

#add optional argument called "--verbose":
parser.add_argument("-v", "--verbose" \
, help="increase output verbosity" \
, action="store_true")

#add optional argument called tf:
parser.add_argument("--tf" \
, help="final simulation time" \
, type=np.float64 \
, default=900.)

###########################################################################

#parse arguments:
args = parser.parse_args()

###########################################################################

print()

#tell what to do with argument called "numToSquare":
if args.verbose :
    print("The square of {0} is {1}" \
    .format(args.numToSquare, args.numToSquare**2))
else :
    print(args.numToSquare**2)

#tell what to do with argument called "stringToPrint":
if args.verbose :
    print("The string you gave as input is: {0}".format(args.stringToPrint))
else :
    print(args.stringToPrint)

#print the value of tf:
print("tf = {0}".format(args.tf))

print()

###########################################################################