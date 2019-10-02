a = {a1: 1,  a2:"2"}
b = {:b1=>1, :b2=>"2"}
c = {"c1"=>1, "c2"=>"2"}
    
puts a
puts b
puts c
puts a.inspect
puts a.to_s
puts a.size
puts a[:a1]
puts a[:a3]

a[:a1] = -1
puts a
puts a.size
puts a[:a1]
puts a[:a3]

a[:a3] = {a3h: 3}
puts a
puts a.size
puts a[:a1]
puts a[:a3]

