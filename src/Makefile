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

SRC =   main.c c_ext.c
GURU= 	console.cu guru.cu alloc.cu static.cu value.cu keyvalue.cu symbol.cu vmalloc.cu \
	 	global.cu vm.cu class.cu load.cu \
	 	c_fixnum.cu
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

c_ext.o: c_ext.c 		c_ext.h
	$(CC) $(CCFLAGS) $<

main.o: main.c
	$(CC) $(CCFLAGS) $<

cuda.o: cuda.cu

console.o: console.cu 	vm_config.h console.h

alloc.o: alloc.cu       vm_config.h guru.h alloc.h

guru.o: guru.cu         vm_config.h console.h guru.h alloc.h

static.o: static.cu	    vm_config.h guru.h

value.o: value.cu       vm_config.h guru.h value.h

keyvalue.o: keyvalue.cu vm_config.h guru.h value.h alloc.h keyvalue.h

symbol.o: symbol.cu 	vm_config.h guru.h value.h alloc.h symbol.h

vmalloc.o: vmalloc.cu 	vm_config.h guru.h value.h alloc.h keyvalue.h symbol.h vmalloc.h

global.o: global.cu	    vm_config.h guru.h value.h vmalloc.h global.h

vm.o: vm.cu             vm_config.h guru.h value.h alloc.h keyvalue.h symbol.h vmalloc.h \
						global.h static.h console.h class.h opcode.h vm.h load.h

c_fixnum.o: c_fixnum.cu vm_config.h guru.h value.h static.h class.h c_fixnum.h

class.o: class.cu	    vm_config.h guru.h value.h alloc.h symbol.h vmalloc.h \
						global.h static.h opcode.h vm.h class.h \
						c_fixnum.h

load.o: load.cu         vm_config.h guru.h alloc.h vmalloc.h vm.h errorcode.h load.h

c_array.o: c_array.c vm_config.h value.h vm.h alloc.h static.h global.h \
  class.h c_array.h console.h

c_string.o: c_string.c vm_config.h value.h alloc.h vm.h static.h global.h \
  class.h c_string.h console.h

c_range.o: c_range.c vm_config.h value.h alloc.h vm.h static.h global.h \
  class.h c_range.h console.h

c_hash.o: c_hash.c vm_config.h value.h vm.h alloc.h static.h global.h \
  class.h c_array.h c_hash.h

clean: 
	$(RM) -R main.o $(LIB) $(OBJS) *~
