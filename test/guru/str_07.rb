#
# strcat unit test
#
A = [ "123", "abc", "4", "X" ]
def sum0
    b = ""
    A.each do |x|
        b += x
        puts b
    end
end
def sum1
    c = ""
    A.each do |x|
        c = "#{c}#{x}"
        puts c
    end
end

sum0()
sum1()
