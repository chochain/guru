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
puts (a.myfunc1 rescue "Err")




