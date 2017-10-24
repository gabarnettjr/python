#I think this only works in Python3.  Not sure why.

import f

print( '\nWelcome to the G A M E of T H R O N E S !' )

player = '';
while player == '' :
	player = input( '\nPlease state your name, your Grace (no more than 10 characters please).\nName: ' )
	if len(player) > 10 :
		player = ''

print( '\nHow vast is your kingdom, ' + player + '?' )

n = ''
while n == '' :
	n = input( 'Enter an integer greater than 3 and less than 30 for the size of the board.\nBoard Size: ' )
	if f.isInt(n) :
		n = int(n)
		if (n<4) | (n>30) :
			n = ''
	else :
		n = ''

A = f.makeMatrix( n )
B = f.blankBoard( n )
C = f.getSolution( A )

qn = [ 0, 0 ]
while qn[0] == 0 :
	qn = f.playGame( A, C, player )
	if qn[1] == 1 :
		A = f.makeMatrix( n )
		C = f.getSolution( A )

print( '\nThank you for Playing!' )
