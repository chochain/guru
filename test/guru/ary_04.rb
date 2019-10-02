a = [0,1,"2"]
b = [3, "4"]
    
puts a
puts a + b
puts a

puts a+=b
puts a
    
a += [5,"6"]
puts a

# a.concat([7, "8"])   # use += (.i.e. OP_ADD instead)
# puts a


