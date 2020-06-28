# Module constant
A = 456

module M
    A = 123
    def a
       A+1
    end 
end

class X
    include M               # M becomes the child of super-class
    p A
    p (a rescue 'err X::a') # only by instantiated object
end

class Y
    extend M                # add f1,f2 as class methods
    p A                     # constant is not extended, use outer scope
    p a                     # 'a' become a class method
end

x = X.new
p x.a+2
p Y.a+3
