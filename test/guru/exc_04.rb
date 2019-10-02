class Tester
    def test(n)
        a = 4/n
    rescue
        -1
    end
end
    
x = Tester.new
puts x.test(2)

