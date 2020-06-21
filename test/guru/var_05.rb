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
x  = MyClass.new

puts c.to_s
puts x.i
puts x.a
    
puts MyClass.a
puts MyClass.h
puts (x.h rescue 'Err xh')
puts (MyClass.i rescue 'Err mi')
