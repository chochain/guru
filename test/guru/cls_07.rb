# multiple inheritance
#
module X
    def x
        1
    end
end

module Y
    include X
    def y
        x+2
    end 
end

class Z
    include Y
end

z = Z.new
p z.x
p z.y
