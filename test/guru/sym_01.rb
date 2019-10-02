a = :symbol123
puts a.to_s
puts a.inspect

b = a.to_sym
puts b.to_s

puts a === b

puts a === :symbol100

