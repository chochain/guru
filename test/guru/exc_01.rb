x = begin
    a = 1
    b = 0
    c = a/b
rescue => e
    ":#{e}"
end
puts x==":0"


