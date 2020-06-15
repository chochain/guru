# optional argumenets
    
def f1(a, b=3, *c)
    p a
    p b
    p c
end

f1(1,2,"X")
f1(1,2,"X",4)
f1(1,2)
f1(1,"X")
f1(1)
    
    
