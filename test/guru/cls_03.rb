# Initialize with parameter
    
class MyClass
  def initialize(n)
    puts "init(#{n})"
  end
  def f1
    puts "f1"
  end
end

class MyClass2 < MyClass
end    

a = MyClass.new(5)
a.f1

b = MyClass2.new(5)
b.f1

class MyClass
    def f2
        p "f2"
    end
end

b.f2
