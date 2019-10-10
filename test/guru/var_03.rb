# class-level var
    
class MyClass
    @@v = 1           # class-level var (belongs to the class)
    def initialize
        @v = 2        # instance-level var
    end
    def get_cv
        @@v
    end
end

x = MyClass.new
puts x.get_cv
