a = [0,1,"2","2"]
b = [3, "4"]
    
puts a
puts a + b
puts a

puts a+=b
puts a
    
a += [5,"6"]
puts a

c = a - ["2", 5]
puts c

#a.concat([7, "8"])         # use += (.i.e. OP_ADD instead)
#puts a


