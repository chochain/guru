# Singleton Method
    
class MyClass
    def f1
        "f1"
    end
end

a = MyClass.new
b = MyClass.new
    
def b.f1
    "b.f1"
end

class << a
    def f2
       "a.f2"
    end
end  
    
p a.f1
p b.f1
p a.f2
p (b.f2 rescue 'err')


    
