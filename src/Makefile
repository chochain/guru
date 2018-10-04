#
# guru  src/Makefile
#
CC=cc
CFLAGS= -Wall -Wpointer-arith -g -DMRBC_DEBUG
NVCC=nvcc
CUDAFLAGS= -dc -std=c++11 -arch=sm_61

LIBS= -lopenblas -lpthread -lcudart -lcublas
LIBDIRS=-L/usr/local/cuda/lib64
INCDIRS=-I/usr/local/cuda/include

SRC = 	guru.cu alloc.cu value.cu keyvalue.cu symbol.cu vmalloc.cu \
	 	global.cu static.cu console.cu vm.cu class.cu load.cu
LIB = 
CUDA = cuda.cu

TARGET = libguru.a

OBJS = $(SRC:.cu=.o) $(LIB:.c=.o) $(CUDA:.cu=.o)

# default to .cu instead of .c
%.o:%.cu;	$(NVCC) $(CUDAFLAGS) $<

all:
	$(MAKE) $(TARGET)

$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) $@ $?

cuda.o: cuda.cu

guru.o: guru.cu         vm_config.h guru.hu

alloc.o: alloc.cu       vm_config.h guru.hu alloc.hu

value.o: value.cu       vm_config.h guru.hu value.hu

keyvalue.o: keyvalue.cu vm_config.h guru.hu alloc.hu keyvalue.hu

symbol.o: symbol.cu 	vm_config.h guru.hu alloc.hu symbol.hu

vmalloc.o: vmalloc.cu 	vm_config.h guru.hu value.hu alloc.hu symbol.hu vmalloc.hu

global.o: global.cu	    vm_config.h guru.hu value.hu vmalloc.hu global.hu

static.o: static.cu	    vm_config.h guru.hu global.hu static.hu

console.o: console.cu 	vm_config.h guru.hu value.hu console.hu

vm.o: vm.cu             vm_config.h guru.hu value.hu alloc.hu global.hu symbol.hu \
						static.hu console.hu class.hu opcode.hu vm.hu

class.o: class.cu	    vm_config.h guru.hu value.hu alloc.hu symbol.hu vmalloc.hu \
						global.hu static.hu opcode.hu vm.hu class.hu 

load.o: load.cu         vm_config.h guru.hu alloc.hu vmalloc.hu vm.hu errorcode.hu load.hu

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
	$(RM) -R $(TARGET) $(OBJS) *~
