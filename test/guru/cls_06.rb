# Module include/extend

module M
    def f1
        "f1 here"
    end
    def f2
        "f2 too"
    end
    p 'here in M'
end

class X
    include M       # add f1, f2 as instance methods
    p (f1 rescue 'err x.f1')
end

class Y
    extend M        # add f1,f2 as class methods
    p f2
end 
    
x = X.new
p x.f1
p (X.f2 rescue 'err X.f2')

y = Y.new
p (y.f1 rescue 'err Y.f1')
p Y.f2
