import f

n = 13;
i = 2;
j = 1;
A = f.makeMatrix( n )
B = f.blankBoard( n )
C = f.getSolution( A )

# ind = f.nearestNeighbors( i, j, n )
# print( ind )

# st = f.playGame( A, C, 'Greg' )
# print( st )

# B = f.showNumber( B, C, i, j, n )
# print( ''.join(B) )
# print( ''.join(C) + '\n\n' )
# print( A )
# print( B )
# print( C )

print( A )
print( ''.join(B) )
print( ''.join(C) )

# B = list( B )
# B = f.showNumber( B, list(C), 3, 0, n )
# print( ''.join(B) )

# B = list( B )
# B[ 3*n+4 + i*(3*n+4) + 5 + j*3 ] = '*'
# B = ''.join(B)
# print( B )

# st = '1234'
# st = list(st)
# print( st[1]=='2' )