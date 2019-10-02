def test(n)
    8 / begin
        a = 4/n
    rescue
        0
    end 
rescue
    -2
end
    
puts test(2)==4
puts test(0)==-2

