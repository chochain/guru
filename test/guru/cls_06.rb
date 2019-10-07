# Module

module MyMod
    def func
        "here"
    end
end

class MyClass
    include MyMod
end
    
a = MyClass.new
puts a.func




