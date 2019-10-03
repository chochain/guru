# exception capture block
    
x = begin
    a = 1
    b = 0
    c = a/b
rescue
    2
end

puts x==2


