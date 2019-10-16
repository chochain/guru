# lambda function
    
def f
    a = 3
    lambda do |b|
        a + b
    end
end

fx = f
puts fx.call(4)
puts fx.call(5)
    
