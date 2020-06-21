# keyword arguments
#
def x(a:, b:2, &c)
    a+b+(c ? c.call : 0)
end

p x(a:1, b:1) { 3 }
p x(a:1) { 3 }
    
