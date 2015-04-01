#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

"""Example usage of an adaline network.

Create an example adaline network using it to decide if a given point is
above the line y = (2x-1)/4
This sounds pretty cheesy, as in order to create the training set, we already
use and algorithm to determine this, but the gag is that the adaline node
is _learning_ to do this decision.

The adaline net we'll create will have two input nodes, recieving the
coordinates of a point that it has to decide on.

Copyright 2004 by Kai Blin
This file is licensed under the GNU GPL, see COPYING for details
$Id: adaline_example.py,v 1.1 2004-01-08 21:55:06 nowhere Exp $
"""
from wfnn.adalinenet import AdalineNet
from wfnn.pattern import Pattern
import random

# we want a set of 500 patterns, type float, with 2 inputs and 1 output
pat = Pattern("float", 2, 1, 500)

# we want an adaline network with two input nodes
net = AdalineNet(2)



for i in range(500):
    x = random.random()*2.0 - 1.0  # adaline network need inputs -1 =< x =< 1
    y = random.random()*2.0 - 1.0
    f = lambda a: (2*a - 1)/4
    res = (y < f(x)) and 1.0 or -1.0  # if y < f(x), the net should eval to 1.0
    pat.setInputs([x, y], i)
    pat.setOutputs([res], i)


# ok, now that the patterns are set up, let's train the network with it

itrn = 0   # counter used to stop training if learing doesn't work
good = 0   # counter used to determine number of correct outputs the network had

while good < 500 and itrn < 1000:
    good = 0
    for i in range(500):
        net.loadInput(pat.getIn(i))   # set the input values
        net.run()                     # run the adaline network
        if net.getValue() != pat.getOut(i)[0]:
            net.train()               # network produced an error, train it
        else:
            good +=1
    itrn += 1

if good == 500:
    print "Congratulations, the network is finished learning."
    print "Saving it to adaline_example.xml"
    outfile = file("adaline_example.xml", 'w')
    net.save(outfile)
    outfile.close()

else:
    print "Learning aborted!"
    sys.exit(0)

# ok, so let's see if it worked.
x = random.random()*2.0 - 1.0  
y = random.random()*2.0 - 1.0
f = lambda a: (2*a - 1)/4
res = (y < f(x)) and 1.0 or -1.0

net.loadInput([x, y])
net.run()
net_res = net.getValue()

print "x = %s, y = %s, result should be: %s" % (x,y,res)
print "The network produced: %s" % net_res