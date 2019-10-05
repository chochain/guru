# Singleton Method
    
class MyClass
    def func
        "method"
    end
end
    
a = MyClass.new
puts a.func

# create a singleton method (in a singleton class)
def a.myfunc  
    "singleton"
end

puts a.func
puts a.myfunc

b = MyClass.new
x = begin
    puts b.func
    b.myfunc
rescue => e
    ":#{e}"
end
puts x[0,18]    # works for both mruby1.4 and ruby2.0
puts x[19,6]

