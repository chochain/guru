#
# guru  src/Makefile
#
CFLAGS = -Wall -Wpointer-arith -g -DMRBC_DEBUG  # -std=c99 -pedantic -pedantic-errors

#COMMON_SRCS = alloc.c class.c console.c global.c keyvalue.c load.c static.c symbol.c value.c vm.c
#RUBY_LIB_SRCS = c_array.c c_hash.c c_numeric.c c_range.c c_string.c mrblib.c
#CUDA_SRC = cuda.cu

COMMON_SRCS = 
RUBY_LIB_SRCS =
CUDA_SRCS = alloc.cu value.cu keyvalue.cu symbol.cu symbol.cu vmalloc.cu \
	 		global.cu static.cu class.cu

TARGET = libguru.a
OBJS = $(COMMON_SRCS:.c=.o) $(RUBY_LIB_SRCS:.c=.o) $(CUDA_SRCS:.cu=.o)

all:
	$(MAKE) $(TARGET)

$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) $@ $?

cuda.o: cuda.cu

alloc.o: alloc.cu       vm_config.h guru.hu alloc.hu

value.o: value.cu       vm_config.h guru.hu value.hu

keyvalue.o: keyvalue.cu vm_config.h guru.hu alloc.hu keyvalue.hu

symbol.o: symbol.cu 	vm_config.h guru.hu alloc.h symbol.hu

vmalloc.o: vmalloc.cu 	vm_config.h guru.hu value.hu alloc.hu symbol.hu vmalloc.hu

global.o: global.cu	    vm_config.h guru.hu value.hu vmalloc.hu global.h

static.o: static.cu	    vm_config.h guru.hu global.hu static.hu

console.o: console.c 	vm_config.h guru.hu value.hu console.hu

vm.o: vm.cu             vm_config.h guru.hu value.hu alloc.hu global.hu symbol.hu \
                        static.hu console.hu class.hu opcode.hu vm.hu

class.o: class.cu	    vm_config.h guru.hu value.hu alloc.hu symbol.hu vmalloc.hu \
                        static.hu cosole.hu class.huvm.h class.h alloc.h static.h \
  global.h symbol.h console.h c_array.h c_hash.h c_numeric.h \
  c_string.h c_range.h

load.o: load.c vm_config.h vm.h value.h load.h errorcode.h alloc.h

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
	@rm -Rf $(TARGET) $(OBJS) *~
