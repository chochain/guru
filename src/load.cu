/*! @file
  @brief
  Guru bytecode loader.

  <pre>
  Copyright (C) 2015-2017 Kyushu Institute of Technology.
  Copyright (C) 2015-2017 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "vm_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "guru.h"
#include "alloc.h"
#include "vmalloc.h"
#include "errorcode.h"
#include "load.h"

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
__GURU__ int load_header(mrbc_vm *vm, const uint8_t **pos)
{
    const uint8_t *p = *pos;

    if (MEMCMP(p, "RITE0004", 8) != 0) {
        vm->error_code = LOAD_FILE_HEADER_ERROR_VERSION;
        return -1;
    }

    /* Ignore CRC */
    /* Ignore size */

    if (MEMCMP(p + 14, "MATZ", 4) != 0) {
        vm->error_code = LOAD_FILE_HEADER_ERROR_MATZ;
        return -1;
    }
    if (MEMCMP(p + 18, "0000", 4) != 0) {
        vm->error_code = LOAD_FILE_HEADER_ERROR_VERSION;
        return -1;
    }
    *pos += 22;
    return 0;
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
  ...		byte codes

  0000_0000	n of pool	(POOL BLOCK)
  (loop n of pool)
  00		type
  0000	length
  ...	pool data

  0000_0000	n of symbol	(SYMS BLOCK)
  (loop n of symbol)
  0000	length
  ...	symbol data
  </pre>
*/
__GURU__ mrbc_irep * load_irep_1(mrbc_vm *vm, const uint8_t **pos)
{
    const uint8_t *p = *pos + 4;			// skip record size

    // new irep
    mrbc_irep *irep = mrbc_irep_alloc(0);
    if (irep==NULL) {
        vm->error_code = LOAD_FILE_IREP_ERROR_ALLOCATION;
        return NULL;
    }

    // nlocals,nregs,rlen
    irep->nlocals = bin_to_uint16(p);	p += 2;
    irep->nregs   = bin_to_uint16(p);	p += 2;
    irep->rlen    = bin_to_uint16(p);	p += 2;
    irep->ilen    = bin_to_uint32(p);	p += 4;

    // padding
    p += (vm->mrb - p) & 0x03;

    // allocate memory for child irep's pointers
    if (irep->rlen) {
        irep->reps = (mrbc_irep **)mrbc_alloc(sizeof(mrbc_irep *) * irep->rlen);
        if (irep->reps==NULL) {
            vm->error_code = LOAD_FILE_IREP_ERROR_ALLOCATION;
            return NULL;
        }
    }

    // ISEQ (code) BLOCK
    irep->code = (uint8_t *)p;
    p += irep->ilen * 4;

    // POOL BLOCK
    irep->plen = bin_to_uint32(p);	p += 4;
    if (irep->plen) {
        irep->pools = (mrbc_object**)mrbc_alloc(sizeof(void*) *irep->plen);
        if (irep->pools==NULL) {
            vm->error_code = LOAD_FILE_IREP_ERROR_ALLOCATION;
            return NULL;
        }
    }

#define MAX_OBJ_SIZE 100

    for (int i = 0; i < irep->plen; i++) {
        int  tt = *p++;
        int  obj_size = bin_to_uint16(p);	p += 2;
        char buf[MAX_OBJ_SIZE];
        mrbc_object *obj = mrbc_obj_alloc(MRBC_TT_EMPTY);
        if (obj==NULL) {
            vm->error_code = LOAD_FILE_IREP_ERROR_ALLOCATION;
            return NULL;
        }
        switch (tt) {
#if MRBC_USE_STRING
        case 0: { // IREP_TT_STRING
            obj->tt = MRBC_TT_STRING;
            obj->str = (char*)p;
        } break;
#endif
        case 1: { // IREP_TT_FIXNUM
            MEMCPY((uint8_t *)buf, p, obj_size);
            buf[obj_size] = '\0';
            
            obj->tt = MRBC_TT_FIXNUM;
            obj->i = ATOL(buf);
        } break;
#if MRBC_USE_FLOAT
        case 2: { // IREP_TT_FLOAT
            MEMCPY((uint8_t *)buf, p, obj_size);
            buf[obj_size] = '\0';
            obj->tt = MRBC_TT_FLOAT;
            obj->d = atof(buf);
        } break;
#endif
        default: break;
        }

        irep->pools[i] = obj;
        p += obj_size;
    }
    // SYMS BLOCK
    irep->ptr_to_sym = (uint8_t*)p;
    int slen = bin_to_uint32(p);		p += 4;
    while (--slen >= 0) {
        int s = bin_to_uint16(p);		p += 2;
        p += s+1;
    }
    *pos = p;
    return irep;
}

//================================================================
/*!@brief
  read all irep section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return       Pointer of allocated mrbc_irep or NULL
*/
__GURU__ mrbc_irep * load_irep_0(mrbc_vm *vm, const uint8_t **pos)
{
    mrbc_irep *irep = load_irep_1(vm, pos);
    if (!irep) return NULL;

    int i;
    for (i = 0; i < irep->rlen; i++) {
        irep->reps[i] = load_irep_0(vm, pos);
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
__GURU__ int load_irep(mrbc_vm *vm, const uint8_t **pos)
{
    const uint8_t *p = *pos + 4;			// 4 = skip "RITE"
    int section_size = bin_to_uint32(p);
    p += 4;
    if (MEMCMP(p, "0000", 4) != 0) {		// rite version
        vm->error_code = LOAD_FILE_IREP_ERROR_VERSION;
        return -1;
    }
    p += 4;
    vm->irep = load_irep_0(vm, &p);
    if (vm->irep==NULL) {
        return -1;
    }

    *pos += section_size;
    return 0;
}

//================================================================
/*!@brief
  Parse LVAR section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of LVAR section.
  @return int	zero if no error.
*/
__GURU__ int load_lvar(mrbc_vm *vm, const uint8_t **pos)
{
    const uint8_t *p = *pos;

    /* size */
    *pos += bin_to_uint32(p+4);

    return 0;
}

//================================================================
/*!@brief
  Load the VM bytecode.

  @param  vm    Pointer to VM.
  @param  ptr	Pointer to bytecode.

*/
__global__ void mrbc_upload_bytecode(mrbc_vm *vm, const uint8_t *ptr)
{
	if (threadIdx.x !=0 || blockIdx.x !=0) return;

    int ret = -1;
    vm->mrb = ptr;

    ret = load_header(vm, &ptr);
    while (ret==0) {
        if (MEMCMP(ptr, "IREP", 4)==0) {
            ret = load_irep(vm, &ptr);
        }
        else if (MEMCMP(ptr, "LVAR", 4)==0) {
            ret = load_lvar(vm, &ptr);
        }
        else if (MEMCMP(ptr, "END\0", 4)==0) {
            break;
        }
    }
}




