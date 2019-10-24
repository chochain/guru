# lambda function

f0 = lambda {|a,b| a+b}
puts f0.call(2,3)
puts f0.call(3,4)
    
def f
    a = 2
    lambda {|b| a+b }     # return this lambda
end
fx = f
puts fx.call(1)
puts fx.call(4)
    
