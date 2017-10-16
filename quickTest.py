def myFunction( a, b ) :
    return a+b, a-b, a*b, a/b

c, d, e, f = myFunction( 4, 3 )
print( 'c = {:f},  d = {:f}, e = {:f}, f = {:f}'.format( c, d, e, f ) )