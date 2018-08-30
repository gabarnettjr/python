import os

testCase = "densityCurrent"

for refinementLevel in range(3):

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
