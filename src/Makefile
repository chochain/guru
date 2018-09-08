#
# guru  src/Makefile
#
CFLAGS = -Wall -Wpointer-arith -g -DMRBC_DEBUG  # -std=c99 -pedantic -pedantic-errors

COMMON_SRCS = alloc.c class.c console.c global.c keyvalue.c load.c static.c symbol.c value.c vm.c
RUBY_LIB_SRCS = c_array.c c_hash.c c_numeric.c c_math.c c_range.c c_string.c mrblib.c
CUDA_SRC = cuda.cu

TARGET = libguru.a
OBJS = $(COMMON_SRCS:.c=.o) $(RUBY_LIB_SRCS:.c=.o) $(CUDA_SRC:.cu=.o)


all:
	$(MAKE) $(TARGET)

$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) $@ $?

cuda.o: cuda.cu

vm.o: vm.c vm_config.h vm.h value.h alloc.h static.h global.h opcode.h \
  class.h symbol.h console.h c_string.h c_range.h c_array.h \
  c_hash.h

value.o: value.c vm_config.h value.h class.h vm.h static.h global.h \
  symbol.h alloc.h c_string.h c_range.h c_array.h c_hash.h

alloc.o: alloc.c vm_config.h alloc.h vm.h value.h console.h 

keyvalue.o: keyvalue.c vm_config.h value.h alloc.h vm.h keyvalue.h

static.o: static.c vm_config.h static.h vm.h value.h global.h class.h \
  symbol.h

global.o: global.c vm_config.h value.h static.h vm.h global.h mrubyc.h \
  errorcode.h alloc.h symbol.h class.h c_array.h c_hash.h \
  c_numeric.h c_range.h c_string.h load.h console.h

class.o: class.c vm_config.h value.h vm.h class.h alloc.h static.h \
  global.h symbol.h console.h c_array.h c_hash.h c_numeric.h \
  c_math.h c_string.h c_range.h

symbol.o: symbol.c vm_config.h value.h vm.h alloc.h static.h global.h \
  class.h symbol.h console.h

load.o: load.c vm_config.h vm.h value.h load.h errorcode.h alloc.h

console.o: console.c vm_config.h console.h

c_array.o: c_array.c vm_config.h value.h vm.h alloc.h static.h global.h \
  class.h c_array.h console.h

c_numeric.o: c_numeric.c vm_config.h c_numeric.h vm.h value.h alloc.h \
  class.h static.h global.h console.h

c_math.o: c_math.c vm_config.h value.h

c_string.o: c_string.c vm_config.h value.h alloc.h vm.h static.h global.h \
  class.h c_string.h console.h

c_range.o: c_range.c vm_config.h value.h alloc.h vm.h static.h global.h \
  class.h c_range.h console.h

c_hash.o: c_hash.c vm_config.h value.h vm.h alloc.h static.h global.h \
  class.h c_array.h c_hash.h


clean:
	@rm -Rf $(TARGET) $(OBJS) *~
