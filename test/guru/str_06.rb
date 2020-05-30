# String methods check
#
# chmop
# chomp!
# dup
# strip

a = " abcdefghijk \n"
b = a.chomp
puts a
puts b
puts a.size
puts b.size

a.chomp!
puts a
puts a.size

c = a.dup
puts c

d = a.strip
puts a
puts d
puts d.size

e = a.strip!
puts e
puts a


