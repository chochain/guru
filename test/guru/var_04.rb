# attr_accessor
    
class MyClass
    attr_accessor :iv  # attribute methods (aka instance-level attributes)
    
    def initialize
        @iv = { a:1 }  # instance-level var
    end
    def get_iv
        @iv
    end
end

x = MyClass.new
puts x.get_iv
puts x.iv
x.iv = [ 1, 2 ]
puts x.get_iv
puts x.iv
