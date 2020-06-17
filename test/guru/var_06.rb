# global/local const
#
V = 5
    
class X
    V = 4
    def f
        3 + V
    end
end

x = X.new
p x.f
p V
