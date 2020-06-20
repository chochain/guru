# Singleton Method
    
class MyClass
    def f1
        "f1"
    end
end

a = MyClass.new
b = MyClass.new
    
def b.f1
    "single"
end
    
puts a.f1
puts b.f1
    
