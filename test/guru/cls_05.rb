# Class Method
    
class MyClass
    def func
        "here"
    end
    def self.myfunc1
        "class method1"
    end
    def self.myfunc2
        "class method2"
    end
end
    
a = MyClass.new
puts a.func
puts MyClass.myfunc2
    
x = begin
    a.myfunc1
rescue => e
    ":#{e}"
end
    
# this works for both mruby1.4+, ruby2.0
puts x[0,18]+x[19,7]




