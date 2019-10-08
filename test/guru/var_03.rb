class MyClass
    @@cv = 1           # class-level var (belongs to the class)
    attr_accessor :iv  # attribute methods (aka instance-level attributes)
    def initialize
        @iv  = 2       # instance-level var
    end
    def get_cv
        @@cv
    end
    def get_iv
        @iv
    end
end

x = MyClass.new
puts x.get_cv
puts x.get_iv
puts x.iv
x.iv = 3
puts x.get_iv
