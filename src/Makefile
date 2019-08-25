#
# guru  src/Makefile
#
CC=gcc
CCFLAGS= -Wall -Wpointer-arith -g -c -DGURU_DEBUG
NVCC=nvcc
CUDAFLAGS= -dc

LIBS= -lopenblas -lpthread -lcudart -lcublas
LIBDIRS=-L/usr/local/cuda/lib64
INCDIRS=-I/usr/local/cuda/include -I./src

SRC =   main.c 
GURU= 	alloc.cu guru.cu static.cu value.cu global.cu \
		store.cu symbol.cu class.cu\
		console.cu sprintf.cu puts.cu load.cu \
		vm.cu opcode.cu object.cu \
		c_fixnum.cu c_string.cu c_range.cu c_array.cu c_hash.cu
GEXT=   c_string.h c_range.h c_array.h c_hash.h
CUDA=   cuda.cu

LIB = 	libguru.a

SRCS = $(SRC:.c=.o)
OBJS = $(GURU:.cu=.o) $(CUDA:.cu=.o)
EXTS = $(GEXT:.cu=.h)

# default to .cu instead of .c
%.o:%.cu;	$(NVCC) $(CUDAFLAGS) $<

all: $(LIB) $(SRCS)
	$(NVCC) -arch=sm_61 -o guru $(SRCS) $(LIB)

lib: $(OBJS)
	$(MAKE) $(LIB)

$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $?

main.o: main.c gurux.h
	$(CC) $(CCFLAGS) $<

cuda.o: cuda.cu

alloc.o: alloc.cu       vm_config.h guru.h alloc.h

guru.o: guru.cu         vm_config.h console.h guru.h gurux.h alloc.h console.h vmx.h

static.o: static.cu	    vm_config.h guru.h

value.o: value.cu       vm_config.h guru.h value.h alloc.h \
						$(EXTS)

global.o: global.cu	    vm_config.h guru.h value.h global.h

store.o: store.cu vm_config.h guru.h alloc.h value.h store.h

symbol.o: symbol.cu 	vm_config.h guru.h alloc.h value.h symbol.h \
						static.h class.h c_array.h

class.o: class.cu	    vm_config.h guru.h value.h alloc.h static.h symbol.h global.h \
						console.h class.h

load.o: load.cu         vm_config.h guru.h alloc.h store.h \
						vm.h errorcode.h load.h

vm.o: vm.cu             vm_config.h guru.h alloc.h value.h static.h store.h \
						console.h opcode.h vm.h load.h

console.o: console.cu 	vm_config.h guru.h value.h console.h

sprintf.o: sprintf.cu   vm_config.h guru.h alloc.h value.h symbol.h console.h sprintf.h

puts.o: puts.cu			vm_config.h guru.h alloc.h value.h symbol.h static.h global.h \
						console.h puts.h $(EXTS)

opcode.o: opcode.cu     vm_config.h guru.h alloc.h value.h static.h \
						symbol.h store.h global.h class.h console.h \
						opcode.h vm.h $(EXTS)

object.o: object.cu	    vm_config.h guru.h value.h alloc.h static.h symbol.h store.h \
						global.h opcode.h vm.h class.h object.h \
						c_fixnum.h $(EXTS)

c_fixnum.o: c_fixnum.cu vm_config.h guru.h value.h static.h \
						class.h c_fixnum.h

c_string.o: c_string.cu vm_config.h guru.h value.h alloc.h static.h symbol.h \
						console.h sprintf.h \
						class.h c_string.h

c_array.o: c_array.cu 	vm_config.h guru.h alloc.h static.h opcode.h object.h \
						console.h errorcode.h \
						class.h c_array.h \
						c_string.h

c_range.o: c_range.cu 	vm_config.h guru.h alloc.h static.h symbol.h class.h \
						console.h c_range.h \
						c_string.h

c_hash.o: c_hash.cu 	vm_config.h guru.h alloc.h static.h \
						class.h c_hash.h \
						c_array.h c_string.h 

clean: 
	$(RM) -R main.o $(LIB) $(OBJS) *~
