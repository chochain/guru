a = [0,1,"2","2"]
b = [3, "4"]
    
puts a.to_s
puts (a + b).to_s
puts a.to_s

puts (a+=b).to_s
puts a.to_s
    
a += [5,"6"]
puts a.to_s

c = a - ["2", 5]
puts c.to_s

#a.concat([7, "8"])         # use += (.i.e. OP_ADD instead)
#puts a


