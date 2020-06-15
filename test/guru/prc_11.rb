# default argumenets
# callback
    
def f1(a, b=2)
    a+b
end

def f2(a=2, b=3)
    a+b
end

def f3(a, b=3, &c)
    a+b+c.call
end 

p f1(1,1)
p f1(1)
p (f1 rescue "err f1")
p f2(1,1)
p f2(1)
p f2
p f3(1,1) { 3 }
p f3(1)   { 30 }
    
    
