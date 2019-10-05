# Singleton Method
    
class MyClass
    def self.myfunc
        "class method"
    end
end
    
puts MyClass.myfunc
    
x = begin
    a = MyClass.new
    a.myfunc
rescue => e
    ":#{e}"
end

puts x[0,18]
puts x[19,6]




