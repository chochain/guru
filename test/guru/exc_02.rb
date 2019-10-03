# rescue in a function
    
def test(n)
    a = 4/n
rescue
    -1
end
    
puts test(2)==2
puts test(0)==-1

