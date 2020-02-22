#
# guru  src/Makefile
#
CC=gcc
CCFLAGS= -Wall -Wpointer-arith -g -c -DGURU_DEBUG
NVCC=nvcc
NVCC_FLAGS= --default-stream per-thread --gpu-architecture=compute_61 --relocatable-device-code=true
CUDAFLAGS= --default-stream per-thread -arch=sm_61 --std=c++11 -DGURU_DEBUG
#CUDAFLAGS= -arch=sm_61 -std=c++11

#LIBS= -lopenblas -lpthread -lcudart -lcublas
LIBS= -lpthread -lcudart 
LIBDIRS=-L/usr/local/cuda/lib64
INCDIRS=-I/usr/local/cuda/include -I./src

SRC =   main.c 
EXT =   c_range.cu c_array.cu c_hash.cu inspect.cu
GURU= 	guru_test.cu vm_debug.cu \
        static.cu global.cu console.cu iter.cu \
		mmu32.cu load.cu \
		ostore.cu symbol.cu class.cu \
		state.cu vm.cu \
		sprintf.cu puts32.cu \
        value.cu refcnt.cu \
		ucode.cu object.cu \
        c_fixnum.cu c_str_utf8.cu \
		$(EXT)
CUDA=   cuda.cu
LIB = 	libguru.a

SRCS = $(SRC:.c=.o)
OBJS = $(GURU:.cu=.o) $(CUDA:.cu=.o)

BASE = vm_config.h gurux.h guru.h mmu.h 
REFS = $(BASE) static.h value.h refcnt.h inspect.h
EXTS = $(EXT:.cu=.h)


# default to .cu instead of .c
%.o:%.cu;	$(NVCC) $(CUDAFLAGS) -dc $<

exe: $(LIB) $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o guru $(SRCS) $(LIB)

lib: $(OBJS)
	$(MAKE) $(LIB)

$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $?

main.o: main.c gurux.h
	$(CC) $(CCFLAGS) $<

cuda.o: cuda.cu

static.o: static.cu	    vm_config.h guru.h static.h

global.o: global.cu	    vm_config.h guru.h global.h

console.o: console.cu 	vm_config.h guru.h console.h

alloc.o: alloc.cu       $(BASE)

mmu32.o: mmu32.cu   	$(BASE) mmu32.h

load.o: load.cu         $(BASE) errorcode.h \
						load.h

guru.o: guru.cu         $(BASE) vmx.h load.h

guru_test.o: guru_test.cu    $(BASE) vmx.h load.h

vm_debug.o: vm_debug.cu $(BASE) state.h symbol.h ucode.h \
                        vm_debug.h

ostore.o: ostore.cu 	$(BASE) \
						ostore.h

symbol.o: symbol.cu 	$(REFS) \
						c_string.h c_array.h \
						symbol.h

class.o: class.cu	    $(REFS) \
						symbol.h global.h \
						class.h

state.o: state.cu		$(REFS) \
						ostore.h symbol.h class.h \
						state.h

vm.o: vm.cu             $(REFS) \
						ostore.h state.h \
						ucode.h c_string.h \
						vm.h

sprintf.o: sprintf.cu   $(REFS) \
						symbol.h console.h \
						sprintf.h

puts.o: puts32.cu		$(REFS) symbol.h global.h \
						$(EXTS) puts.h

value.o: value.cu       $(REFS) \
						$(EXTS)

refcnt.o: refcnt.cu     $(REFS) \
						$(EXTS)

ucode.o: ucode.cu       $(REFS) \
						symbol.h ostore.h global.h class.h \
						vm.h ucode.h \
						$(EXTS)

c_fixnum.o: c_fixnum.cu $(REFS) \
						c_string.h c_fixnum.h

object.o: object.cu	    $(REFS) symbol.h ostore.h \
						global.h ucode.h vm.h class.h object.h \
						c_fixnum.h $(EXTS)

c_str_utf8.o: c_str_utf8.cu $(REFS) symbol.h c_string.h

c_array.o: c_array.cu 	$(REFS) c_array.h

c_hash.o: c_hash.cu 	$(REFS) c_array.h c_hash.h

c_range.o: c_range.cu 	$(REFS) object.h c_range.h

inspect.o: inspect.cu   $(REFS) object.h symbol.h \
                        c_fixnum.h $(EXTS)

iter.o: iter.cu         $(REFS) \
						iter.h

clean: 
	$(RM) -R main.o $(LIB) $(OBJS) *~
