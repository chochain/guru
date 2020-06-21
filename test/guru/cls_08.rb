# module with new method
#
module X
    def x
        1
    end
end

class Y
    include X
end

y = Y.new
p y.x

module X
    def x2
        2
    end
end

p y.x2


