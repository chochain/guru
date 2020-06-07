# attr_accessor
    
class MyClass
    attr_accessor :iv  # attribute methods (aka instance-level attributes)
    
    def initialize
        @iv = { a:1 }  # instance-level var
        @ix = 2
    end
    def get_iv
        @iv
    end
    def get_ix
        @ix
    end 
end

x = MyClass.new
puts x.get_iv
puts x.iv
x.iv = [ 1, 2 ]
puts x.get_iv
puts x.iv
puts x.get_ix
puts (x.ix rescue "err")
    
