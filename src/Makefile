#
# guru  src/Makefile
#
CC=cc
CFLAGS= -Wall -Wpointer-arith -g -c -DMRBC_DEBUG
NVCC=nvcc
CUDAFLAGS= -dc

LIBS= -lopenblas -lpthread -lcudart -lcublas
LIBDIRS=-L/usr/local/cuda/lib64
INCDIRS=-I/usr/local/cuda/include -I./src

SRC =   main.c 
GURU= 	alloc.cu guru.cu static.cu value.cu global.cu \
		keyvalue.cu instance.cu symbol.cu \
		console.cu sprintf.cu puts.cu \
		load.cu vm.cu class.cu object.cu \
	 	object.cu c_fixnum.cu c_string.cu
CUDA=   cuda.cu

LIB = 	libguru.a

SRCS = $(SRC:.c=.o)
OBJS = $(GURU:.cu=.o) $(CUDA:.cu=.o)

# default to .cu instead of .c
%.o:%.cu;	$(NVCC) $(CUDAFLAGS) $<

all: $(LIB) $(SRCS)
	$(NVCC) -arch=sm_61 -o guru $(SRCS) $(LIB)

lib: $(OBJS)
	$(MAKE) $(LIB)

$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $?

main.o: main.c
	$(CC) $(CCFLAGS) $<

cuda.o: cuda.cu

alloc.o: alloc.cu       vm_config.h guru.h alloc.h

guru.o: guru.cu         vm_config.h console.h guru.h alloc.h

static.o: static.cu	    vm_config.h guru.h

value.o: value.cu       vm_config.h guru.h value.h

global.o: global.cu	    vm_config.h guru.h value.h global.h

keyvalue.o: keyvalue.cu vm_config.h guru.h alloc.h value.h keyvalue.h

instance.o: instance.cu vm_config.h guru.h alloc.h value.h keyvalue.h instance.h

symbol.o: symbol.cu 	vm_config.h guru.h alloc.h value.h symbol.h

console.o: console.cu 	vm_config.h console.h

sprintf.o: sprintf.cu   vm_config.h guru.h alloc.h value.h symbol.h console.h sprintf.h

puts.o: puts.cu			vm_config.h guru.h alloc.h value.h symbol.h static.h global.h \
						console.h puts.h

load.o: load.cu         vm_config.h guru.h alloc.h instance.h vm.h errorcode.h load.h \
						vm.h

vm.o: vm.cu             vm_config.h guru.h alloc.h value.h static.h keyvalue.h symbol.h \
						instance.h global.h console.h class.h opcode.h vm.h load.h

class.o: class.cu	    vm_config.h guru.h value.h alloc.h static.h symbol.h global.h \
						vm.h

object.o: object.cu	    vm_config.h guru.h value.h alloc.h static.h symbol.h instance.h \
						global.h opcode.h vm.h class.h object.h \
						c_fixnum.h

c_fixnum.o: c_fixnum.cu vm_config.h guru.h value.h static.h class.h c_fixnum.h

c_string.o: c_string.cu vm_config.h guru.h value.h alloc.h static.h symbol.h class.h \
						console.h sprintf.h c_string.h 
						
c_array.o: c_array.c vm_config.h value.h vm.h alloc.h static.h global.h \
  class.h c_array.h console.h

c_range.o: c_range.c vm_config.h value.h alloc.h vm.h static.h global.h \
  class.h c_range.h console.h

c_hash.o: c_hash.c vm_config.h value.h vm.h alloc.h static.h global.h \
  class.h c_array.h c_hash.h

clean: 
	$(RM) -R main.o $(LIB) $(OBJS) *~
