# OP_ARYCAT
#
a = [ 1, "A", 2 ]
b = [ 3, "B", 4, "C" ]
c = [ 5, "D" ]
    
def x(p, q, r)
    puts "#{p},#{q}"
    r
end
    
def y(p, q, *r)
    puts "#{p},#{q}"
    r
end

p x(*a)
p (x(*b) rescue "err b")
p (x(*c) rescue "err c")
p y(*a)
p y(*b)
p y(*c)
