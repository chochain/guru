#
# guru  src/Makefile
#
CC=cc
CFLAGS= -Wall -Wpointer-arith -g -c -DMRBC_DEBUG
NVCC=nvcc
CUDAFLAGS= -dc -std=c++11 -arch=sm_61

LIBS= -lopenblas -lpthread -lcudart -lcublas
LIBDIRS=-L/usr/local/cuda/lib64
INCDIRS=-I/usr/local/cuda/include

SRC =   main.c
GURU= 	guru.cu alloc.cu value.cu keyvalue.cu symbol.cu vmalloc.cu \
	 	global.cu static.cu console.cu vm.cu class.cu load.cu
CUDA=   cuda.cu

LIB = 	libguru.a

SRCS = $(SRC:.c=.o)
OBJS = $(GURU:.cu=.o) $(CUDA:.cu=.o)

# default to .cu instead of .c
%.o:%.cu;	$(NVCC) $(CUDAFLAGS) $<

all: $(LIB)
	$(NVCC) -arch=sm_61 -o guru $(LIB)

lib: $(OBJS)
	$(MAKE) $(LIB)

$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $?

main.o: main.c
	$(CC) $(CCFLAGS) $<

cuda.o: cuda.cu

guru.o: guru.cu         vm_config.h guru.h

alloc.o: alloc.cu       vm_config.h guru.h alloc.h

value.o: value.cu       vm_config.h guru.h value.h

keyvalue.o: keyvalue.cu vm_config.h guru.h alloc.h keyvalue.h

symbol.o: symbol.cu 	vm_config.h guru.h alloc.h symbol.h

vmalloc.o: vmalloc.cu 	vm_config.h guru.h value.h alloc.h symbol.h vmalloc.h

global.o: global.cu	    vm_config.h guru.h value.h vmalloc.h global.h

static.o: static.cu	    vm_config.h guru.h global.h static.h

console.o: console.cu 	vm_config.h guru.h value.h console.h

vm.o: vm.cu             vm_config.h guru.h value.h alloc.h global.h symbol.h \
						static.h console.h class.h opcode.h vm.h

class.o: class.cu	    vm_config.h guru.h value.h alloc.h symbol.h vmalloc.h \
						global.h static.h opcode.h vm.h class.h

load.o: load.cu         vm_config.h guru.h alloc.h vmalloc.h vm.h errorcode.h load.h

c_array.o: c_array.c vm_config.h value.h vm.h alloc.h static.h global.h \
  class.h c_array.h console.h

c_numeric.o: c_numeric.c vm_config.h c_numeric.h vm.h value.h alloc.h \
  class.h static.h global.h console.h

c_string.o: c_string.c vm_config.h value.h alloc.h vm.h static.h global.h \
  class.h c_string.h console.h

c_range.o: c_range.c vm_config.h value.h alloc.h vm.h static.h global.h \
  class.h c_range.h console.h

c_hash.o: c_hash.c vm_config.h value.h vm.h alloc.h static.h global.h \
  class.h c_array.h c_hash.h

clean: 
	$(RM) -R main.o $(LIB) $(OBJS) *~
