# raise exception

class Tester
    MAX = 3
    def test(n)
        i = 0
        while i<MAX
            raise "Err" if i==n
            i += 1
        end
        "OK"
    rescue => e
        ":#{e}"
    end
end
    
x = Tester.new
puts x.test(4)=="OK"
puts x.test(2)==":Err"

