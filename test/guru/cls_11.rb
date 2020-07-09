#
# module/class namespace
#
module M
    class A
        def a
            123
        end
    end
    def m
        o = A.new
        p o.a
    end 
end

x = M::A.new
p x.a+100
p (A.new rescue 'err ::A')
    
class B
    extend M
    self.m
end

p (B::A.new rescue 'err B::A')

