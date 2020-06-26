# module with new method
#
module M
    def f
        1
    end
end

class Y
    include M
end

y = Y.new
p y.f

module M
    def f2
        2
    end
end

p y.f2


