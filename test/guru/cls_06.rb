# Module include/extend

module M
    def f1
        "f1 here"
    end
    def f2
        "f2 too"
    end
end

class X
    include M       # add f1, f2 as instance methods
end

class Y
    extend M        # add f1,f2 as class methods
end 
    
x = X.new
p x.f1
p (X.f2 rescue 'err X')

y = Y.new
p (y.f1 rescue 'err Y')
p Y.f2
