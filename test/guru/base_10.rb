#
# map/collect
#
x = [ 1, 2, 3 ].map {|i| i*i }
y = { a:1, b:2 }.map {|k,v| "#{k}=>#{v}" }
puts x
puts y

