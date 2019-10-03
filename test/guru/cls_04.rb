# Class puts
    
class MyClass
  def initialize(n)
    puts "class init"
    puts n
  end
  def func
    puts "class func"
  end
end

class MyClass2 < MyClass
end    

a = MyClass.new(5)
a.func
puts a
puts a.to_s

b = MyClass2.new(5)
b.func
puts b
puts b.to_s
