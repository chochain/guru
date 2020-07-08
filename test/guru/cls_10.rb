# subclass case
#   OP_OCLASS, OP_GETMCNST
#   obj#const_set, obj#const_get direct call
#
::Object.const_set :A, Class.new  # add constant which is a new class named A to Object's cache
p ::A
    
::A.const_set :B, Class.new       # add a constant which is a new class named B to A's cache
p A::B
    
::A.const_set :C, Module.new      # add another constant that is a new Module named C to A's cache
p A::C
