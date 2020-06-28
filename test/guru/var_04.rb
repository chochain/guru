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
p x.get_iv
p x.iv
x.iv = [ 1, 2 ]
p x.get_iv
p x.iv
p x.get_ix
p (x.ix rescue "err")
    
