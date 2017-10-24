import numpy as np

def makeMatrix( n ) :
	#Make the matrix A with ones (kings) in random places
	numKings = round( 3*n**2/20 )                 #about 3 kings per 20 towns
	A = np.zeros( (n,n) )
	while np.count_nonzero(A) < numKings :
		A[ np.random.randint(0,n), np.random.randint(0,n) ] = 1
	return A

def blankBoard( n ) :
	#Make the blank game board
	B = '\n     '
	for j in range(n) :
		if j < 10 :
			B = B + ' ' + str(j) + ' '
		else :
			B = B + ' ' + str(j)
	tmp = '     '
	for i in range(n) :
		tmp = tmp + '---'
	B = B + '\n' + tmp
	for i in range(n) :
		if i < 10 :
			tmp = '  ' + str(i) + ' |'
		else :
			tmp = ' '  + str(i) + ' |'
		for j in range(n) :
			tmp = tmp + ' . '
		B = B + '\n' + tmp
	return list(B)

def nearestNeighbors( i, j, n ) :
	#find the index of adjacent squares to location (i,j)
	count = 0
	for ii in range(n) :
		for jj in range(n) :
			if (abs(ii-i)<=1) & (abs(jj-j)<=1) & ((ii!=i)|(jj!=j)) :
				if count == 0 :
					ind = [ii,jj]
				else :
					ind = np.vstack(( ind, [ii,jj] ))
				count = count + 1
	return ind

def specialIndex( i, j, n ) :
	return (i+2)*(3*n+6) + 7+j*3

def getSolution( A ) :
	n = np.shape(A)[0]
	C = blankBoard( n )
	for i in range(n) :
		for j in range(n) :
			if A[i,j] == 1 :
				C[ specialIndex(i,j,n) ] = '*'
			else :
				ind = nearestNeighbors( i, j, n )
				count = 0
				for k in range( np.shape(ind)[0] ) :
					if A[ ind[k,0], ind[k,1] ] == 1 :
						count = count + 1
				C[ specialIndex(i,j,n) ] = str(count)
	return C

def showNumber( B, C, i, j, n ) :
	if B[ specialIndex(i,j,n) ] == '.' :
		B[ specialIndex(i,j,n) ] = C[ specialIndex(i,j,n) ]
		if B[ specialIndex(i,j,n) ] == '0' :
			ind = nearestNeighbors( i, j, n )
			for k in range( np.shape(ind)[0] ) :
				B = showNumber( B, C, ind[k,0], ind[k,1], n )
	return B

def isInt(s) :
    try :
        int(s)
        return True
    except ValueError :
        return False

def playGame( A, C, player ) :
	n = np.shape(A)[0]
	numKings = np.count_nonzero( A )
	numKnights = numKings
	print( '\nThe Game begins!  You have ' + str(numKnights) + ' knights, and there are ' + str(numKings) + ' kings to find.\n' )
	B = blankBoard( n )
	print( ''.join(B) + '\n\n' )
	kingsFound = 0
	nMoves = 0
	while (kingsFound<numKings) & (numKnights>0) :
		ind = -1
		while ind == -1 :
			st = input( "Enter a row and column of an unexplored town, separated by a space.\n'row' space 'column': " )
			for k in range( len(st) ) :
				if st[k] == ' ' :
					ind = k
			if ( ind != -1 ) :
				i = st[0:ind]
				j = st[ind+1:len(st)]
				if ( isInt(i) & isInt(j) ) :
					i = int( i )
					j = int( j )
					if (i<0) | (i>n-1) | (j<0) | (j>n-1) :
						ind = -1
					elif B[specialIndex(i,j,n)] != '.' :
						ind = -1
				else :
					ind = -1
		nMoves = nMoves + 1
		if A[i,j] == 1 :
			kingsFound = kingsFound + 1
			numKnights = numKnights + 2
			print( '\nYou found a king!  You have ' + str(numKnights) + ' knights and there are ' + str(numKings-kingsFound) + ' kings left.\n' )
			B[ specialIndex(i,j,n) ] = '*'
		else :
			numKnights = numKnights - 1
			print( "\nYou Didn't find a king.  You have " + str(numKnights) + ' knights and there are ' + str(numKings-kingsFound) + ' kings left.\n' )
			B = showNumber( B, C, i, j, n )
		print( ''.join(B) + '\n\n')
	if kingsFound == numKings :
		print( 'You win!  It took you ' + str(nMoves) + ' moves to win.  Here is the full board:\n' )
		print( ''.join(C) )
		st = '';
		while (st!='y') & (st!='n') :
			st = input( '\n\nWould you like to play again (y/n)?\n' )
		if st == 'y' :
			qn = [ 0, 1 ]
		else :
			qn = [ 1, 0 ]
	else :
		print( 'You lose!  You have no knights left to find the remaining kings!\n' )
		st = ''
		while (st!='y') & (st!='n') :
			st = input( 'Would you like to play again on the SAME board (y/n)?\n' )
		if st == 'y' :
			qn = [ 0, 0 ]
		else :
			st = ''
			while (st!='y') & (st!='n') :
				st = input( 'Would you like to play again on a DIFFERENT board (y/n)?\n' )
			if st == 'y' :
				qn = [ 0, 1 ]
			else :
				qn = [ 1, 0 ]
	return qn
