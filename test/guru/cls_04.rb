# Singleton Method
    
class MyClass
    def myfunc
        "here"
    end
end

a = MyClass.new
b = MyClass.new
    
def b.myfunc
    "single"
end
    
puts a.myfunc
puts b.myfunc
    
