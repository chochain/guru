# optional argumenets, with
# rest argument
    
def f1(a, b=3, c=4, *d)
    p a+b+c
    d
end

p f1(1,2,3,"X")
p f1(1,2,3,"X",4)
p f1(1,2,3)
p f1(1,2)
p f1(1)
    
    
