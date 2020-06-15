# Array.each
#
def eachputs(ary)
    ary.each do |i|
        puts i
    end
end

a = [0, 1, "X"]
b = a + [3, "Y"]

eachputs(a)
eachputs(b)


