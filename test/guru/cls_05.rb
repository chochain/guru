# Class Method
    
class MyClass
    def f
        "here"
    end
    def self.f1
        "class f1"
    end
    def self.f2
        "class f2"
    end
end
    
a = MyClass.new
puts a.f
puts MyClass.f2
puts (a.f1 rescue 'err f1')
    




