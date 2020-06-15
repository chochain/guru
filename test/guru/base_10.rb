#
# map/collect
#
x = [ 1, 2, 3 ].map {|i| i*i }
p x
y = (4..5).map {|i| i+i }
p y
z = { a:1, b:2 }.map {|k,v| "#{k}=>#{v}" }
p z

