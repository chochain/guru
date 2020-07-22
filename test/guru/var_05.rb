# instance var, class var
    
class MyClass
    class << self; attr_accessor :a, :h; end
    attr_accessor :i, :a

    def initialize
        @i = 1
        @a = [1,2]
    end
    def self.init
        @a, @h = [3, "X"], {a:4, b:"Y"}
    end
end

c  = MyClass.init
p c.to_s
p MyClass.a
p MyClass.h
    
x  = MyClass.new
p x.i
p x.a
    
p (x.h rescue 'Err xh')
p (MyClass.i rescue 'Err mi')
