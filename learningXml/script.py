
import sys
sys.path.append("../site-packages/gab")
import myXml

x = ""
with open("test.xml", "r") as f:
    for line in f:
        x = x + line.strip()

y = myXml.getVal(x, "<Years>")
print(y)

