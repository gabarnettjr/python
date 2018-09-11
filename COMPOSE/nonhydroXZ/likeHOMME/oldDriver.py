import os
import sys

testCase = sys.argv[1]

for refinementLevel in range(3,4):

    s = testCase + " height vEul " + str(refinementLevel)
    print("\n\n\n" + s + "\n")
    os.system("python oldMain.py " + s)

    s = testCase + " height vLag " + str(refinementLevel)
    print("\n\n\n" + s + "\n")
    os.system("python oldMain.py " + s)
    
    s = testCase + " pressure vEul " + str(refinementLevel)
    print("\n\n\n" + s + "\n")
    os.system("python oldMain.py " + s)

    s = testCase + " pressure vLag " + str(refinementLevel)
    print("\n\n\n" + s + "\n")
    os.system("python oldMain.py " + s)
