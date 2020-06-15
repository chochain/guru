# closure
#
def x(a, b, &c)
    puts a + b
    c.call
end

x(1, 2) { puts 4 }
