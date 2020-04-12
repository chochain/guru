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
CORE= 	util.cu static.cu global.cu mmu32.cu symbol.cu \
		base.cu ostore.cu iter.cu
EXT =   c_fixnum.cu c_range.cu c_string.cu c_array.cu c_hash.cu 
GURU= 	$(CORE) $(EXT) \
        guru_test.cu load.cu class.cu state.cu \
		object.cu ucode.cu vm.cu \
		inspect.cu debug.cu \
        console.cu sprintf.cu puts32.cu
CUDA=   cuda.cu
LIB = 	libguru.a

SRCS = $(SRC:.c=.o)
OBJS = $(GURU:.cu=.o) $(CUDA:.cu=.o)

_CORE = vm_config.h gurux.h guru.h util.h mmu.h base.h 
_EXT  = $(EXT:.cu=.h) class.h inspect.h
_GURU = $(_EXT) state.h object.h ucode.h vm.h 


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

util.o: util.cu			vm_config.h

static.o: static.cu	    vm_config.h guru.h static.h

global.o: global.cu	    vm_config.h guru.h global.h

mmu32.o: mmu32.cu   	$(_CORE)

symbol.o: symbol.cu 	$(_CORE) symbol.h

base.o: base.cu     	$(_CORE)

ostore.o: ostore.cu 	$(_CORE) ostore.h

iter.o: iter.cu         $(_CORE) iter.h

c_fixnum.o: c_fixnum.cu $(_EXT) c_fixnum.h

c_range.o: c_range.cu 	$(_EXT) object.h c_range.h

c_string.o: c_string.cu $(_EXT) c_string.h

c_array.o: c_array.cu 	$(_EXT) c_array.h

c_hash.o: c_hash.cu 	$(_EXT) c_array.h c_hash.h

guru_test.o: guru_test.cu    $(_CORE)

guru.o: guru.cu         $(_CORE) vmx.h load.h

class.o: class.cu	    $(_CORE) class.h

state.o: state.cu		$(_CORE) class.h state.h

load.o: load.cu         $(_CORE) state.h errorcode.h load.h

object.o: object.cu	    $(_GURU) puts.h

ucode.o: ucode.cu       $(_GURU)

vm.o: vm.cu             $(_GURU) vmx.h debug.h

inspect.o: inspect.cu   $(_EXT)

debug.o: debug.cu 		$(_CORE) state.h ucode.h debug.h

console.o: console.cu 	vm_config.h guru.h console.h

sprintf.o: sprintf.cu   $(_CORE) console.h sprintf.h

puts.o: puts32.cu		$(_EXT) puts.h

clean: 
	$(RM) -R main.o $(LIB) $(OBJS) *~
