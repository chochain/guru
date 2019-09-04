/*! @file
  @brief
  GURU bytecode loader (IREP code parse by CUDA device directly).

  alternatively, load.cu can be used for host built image (then passed into device for execution)
  <pre>
  Copyright (C) 2019- Greeni

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <assert.h>
#include "vm_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "guru.h"
#include "value.h"
#include "alloc.h"
#include "errorcode.h"
#include "vm.h"
#include "load.h"

#if !GURU_HOST_IMAGE
//================================================================
/*!@brief
  Parse header section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of RITE header.
  @return int	zero if no error.

  <pre>
  Structure
  "RITE"	identifier
  "0004"	version
  0000		CRC
  0000_0000	total size
  "MATZ"	compiler name
  "0000"	compiler version
  </pre>
*/
__GURU__ int
_load_header(U8P *pos)
{
    U8P p = *pos;

    if (MEMCMP(p, "RITE0004", 8) != 0) {
        return LOAD_FILE_HEADER_ERROR_VERSION;
    }

    /* Ignore CRC */
    /* Ignore size */

    if (MEMCMP(p + 14, "MATZ", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_MATZ;
    }
    if (MEMCMP(p + 18, "0000", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_VERSION;
    }
    *pos += 22;

    return NO_ERROR;
}

//================================================================
/*!@brief
  read one irep section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return       Pointer of allocated mrbc_irep or NULL

  <pre>
  (loop n of child irep bellow)
  0000_0000	record size
  0000		n of local variable
  0000		n of register
  0000		n of child irep

  0000_0000	n of byte code  (ISEQ BLOCK)
  ...		byte codes		(padded to 4-byte align)

  0000_0000	n of pool		(POOL BLOCK)
  (loop n of pool)
  00		type
  0000		length
  ...		pool data

  0000_0000	n of symbol		(SYMS BLOCK)
  (loop n of symbol)
  0000		length
  ...		symbol data
  </pre>
*/
__GURU__ U32
_load_irep_1(mrbc_irep *irep, U8P *pos)
{
    U8 * p = U8PADD(*pos, 4);								// skip "IREP"

    // nlocals,nregs,rlen
    irep->nlv  = _bin_to_u16(p);	p += sizeof(U16);		// number of local variables
    irep->nreg = _bin_to_u16(p);	p += sizeof(U16);		// number of registers used
    irep->rlen = _bin_to_u16(p);	p += sizeof(U16);		// number of child IREP blocks
    irep->ilen = _bin_to_u32(p);	p += sizeof(U32);		// ISEQ (bytecodes) length

    p += U8POFF(irep, p) & 0x03;							// 32-bit align code pointer

    irep->iseq = (U32P)p;			p += irep->ilen * sizeof(U32);		// ISEQ (code) block
    irep->plen = _bin_to_u32(p);	p += sizeof(U32);					// POOL block

    if (irep->rlen) {					// allocate child irep's pointers (later filled by _load_irep_0)
        irep->list = (mrbc_irep **)guru_alloc(sizeof(mrbc_irep *) * irep->rlen);
        if (irep->list==NULL) {
            return LOAD_FILE_IREP_ERROR_ALLOCATION;
        }
    }
    if (irep->plen) {					// allocate pool of object pointers
        irep->pool = (mrbc_object**)guru_alloc(sizeof(void*) * irep->plen);
        if (irep->pool==NULL) {
            return LOAD_FILE_IREP_ERROR_ALLOCATION;
        }
    }

    for (U32 i=0; i < irep->plen; i++) {		// build object pool
        U32 tt = *p++;
        U32 obj_size = _bin_to_u16(p);	p += sizeof(U16);
        U8  buf[64+2];

        mrbc_object *obj = (mrbc_object *)guru_alloc(sizeof(mrbc_object));
        if (obj==NULL) {
            return LOAD_FILE_IREP_ERROR_ALLOCATION;
        }
        switch (tt) {
        case 0: { 	// IREP_GT_STRING
            obj->gt  = GT_STR;
            obj->sym = p;
        } break;
        case 1: { 	// IREP_GT_FIXNUM
            MEMCPY(buf, p, obj_size);
            buf[obj_size] = '\0';

            obj->gt = GT_INT;
            obj->i = (int)ATOI(buf);
        } break;
#if GURU_USE_FLOAT
        case 2: { 	// IREP_GT_FLOAT
            MEMCPY(buf, p, obj_size);
            buf[obj_size] = '\0';
            obj->gt = GT_FLOAT;
            obj->f  = ATOF(buf);
        } break;
#endif
        default:
        	obj->gt = GT_EMPTY;	// other object are not supported yet
        	break;
        }
        irep->pool[i] = obj;			// stick it into object pool array
        p += obj_size;
    }

    // SYMS BLOCK
    irep->sym = p;
    int sym_cnt =
    		irep->slen = _bin_to_u32(p);	p += sizeof(U32);
    for (U32 i=0; i<sym_cnt; i++) {
        int len = _bin_to_u16(p);			p += sizeof(U16)+len+1;    // symbol_len+'\0'
    }
    *pos = p;

    return NO_ERROR;
}

//================================================================
/*!@brief
  read all irep section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return       Pointer of allocated mrbc_irep or NULL
*/
__GURU__ mrbc_irep*
_load_irep_0(U8P *pos)
{
    // new irep
    mrbc_irep *irep = (mrbc_irep *)guru_alloc(sizeof(mrbc_irep));
    if (irep==NULL) {
        return NULL;
    }
    int ret = _load_irep_1(irep, pos);		// populate content of current IREP
    if (ret != NO_ERROR) {
    	guru_free(irep);
    	return NULL;
    }
    // recursively create the child irep tree
    for (U32 i=0; i<irep->rlen; i++) {
        irep->list[i] = _load_irep_0(pos);
    }
    return irep;
}

//================================================================
/*!@brief
  Parse IREP section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return int	zero if no error.

  <pre>
  Structure
  "IREP"	section identifier
  0000_0000	section size
  "0000"	rite version
  </pre>
*/
__GURU__ int
_load_irep(mrbc_vm *vm, U8P *pos)
{
    U8 * p = U8PADD(*pos, 4);							// 4 = skip "IREP"
    int   sec_size = _bin_to_u32(p); p += sizeof(U32);

    if (MEMCMP(p, "0000", 4) != 0) {					// IREP version
		return LOAD_FILE_IREP_ERROR_VERSION;
    }
    p += 4;												// 4 = skip "0000"

    vm->irep = _load_irep_0(&p);						// recursively load irep tree
    if (vm->irep==NULL) {
        return LOAD_FILE_IREP_ERROR_ALLOCATION;
    }

    *pos += sec_size;
    return NO_ERROR;
}

//================================================================
/*!@brief
  Parse LVAR section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of LVAR section.
  @return int	zero if no error.
*/
__GURU__ int
_load_lvar(mrbc_vm *vm, U8P *pos)
{
    U8P p = *pos;

    /* size */
    *pos += _bin_to_u32(p+sizeof(U32));

    return NO_ERROR;
}

//================================================================
/*!@brief
  Load the VM bytecode.

  @param  vm    Pointer to VM.
  @param  ptr	Pointer to bytecode.

*/
__GPU__ void
mrbc_parse_bytecode(mrbc_vm *vm, U8P ptr)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

    int ret = _load_header(&ptr);

    while (ret==NO_ERROR) {
        if (MEMCMP(ptr, "IREP", 4)==0) {
        	ret = _load_irep(vm, &ptr);
        }
        else if (MEMCMP(ptr, "LVAR", 4)==0) {
            ret = _load_lvar(vm, &ptr);
        }
        else if (MEMCMP(ptr, "END\0", 4)==0) {
            break;
        }
    }
    __syncthreads();
}

__HOST__ void
_show_decoder(mrbc_vm *vm)
{
	U16  pc   = vm->state->pc;
	U32P iseq = vm->state->irep->iseq;
	U16  opid = (*(iseq + pc) >> 24) & 0x7f;
	GV   *v   = vm->regfile;
	const U8P opc  = _opcode[GET_OPCODE(opid)];

	int last=0;
	for (U32 i=0; i<MAX_REGS_SIZE; i++, v++) {
		if (v->gt==GT_EMPTY) continue;
		last=i;
	}

	int lvl=0;
	guru_state *st = vm->state;
	while (st->prev != NULL) {
		st = st->prev;
		lvl++;
	}

	v = vm->regfile;	// rewind
	int s[8];
	guru_malloc_stat(s);
	printf("%c%-4d%-8s%4d[", 'a'+lvl, pc, opc, s[3]);

	for (U32 i=0; i<=last; i++, v++) {
		printf("%2d.%s", i, _vtype[v->gt]);
	    if (v->gt >= GT_OBJ) printf("_%d", v->self->refc);
	    printf(" ");
    }
	printf("]\n");
}

__HOST__ void
mrbc_show_irep(mrbc_irep *irep)
{
	printf("\tnregs=%d, nlocals=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			irep->nreg, irep->nlv, irep->plen, irep->slen, irep->rlen, irep->ilen);

	// dump all children ireps
	for (U32 i=0; i<irep->rlen; i++) {
		mrbc_show_irep(irep->list[i]);
	}
}
#endif	// !GURU_HOST_IMAGE
