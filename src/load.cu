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
#include "value.h"
#include "alloc.h"
#include "errorcode.h"
#include "vm.h"
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
__GURU__ int
_load_header(const uint8_t **pos)
{
    const uint8_t *p = *pos;

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
__GURU__ int
_load_irep_1(mrbc_irep *irep, const uint8_t **pos)
{
    const uint8_t *p = *pos + 4;		// skip "IREP"

    // nlocals,nregs,rlen
    irep->nlv  = _bin_to_uint16(p);	p += sizeof(uint16_t);
    irep->nreg = _bin_to_uint16(p);	p += sizeof(uint16_t);
    irep->rlen = _bin_to_uint16(p);	p += sizeof(uint16_t);
    irep->ilen = _bin_to_uint32(p);	p += sizeof(uint32_t);

    p += ((uint8_t *)irep - p) & 0x03;	// 32-bit align code pointer

    irep->iseq = (uint8_t *)p;			p += irep->ilen * sizeof(uint32_t);		// ISEQ (code) block
    irep->plen = _bin_to_uint32(p);		p += sizeof(uint32_t);					// POOL block

    // allocate memory for child irep's pointers
    if (irep->rlen) {
        irep->irep_list = (mrbc_irep **)mrbc_alloc(sizeof(mrbc_irep *) * irep->rlen);
        if (irep->irep_list==NULL) {
            return LOAD_FILE_IREP_ERROR_ALLOCATION;
        }
    }
    if (irep->plen) {
        irep->pool = (mrbc_object**)mrbc_alloc(sizeof(void*) *irep->plen);
        if (irep->pool==NULL) {
            return LOAD_FILE_IREP_ERROR_ALLOCATION;
        }
    }

    for (int i = 0; i < irep->plen; i++) {
        int  tt = *p++;
        int  obj_size = _bin_to_uint16(p);	p += sizeof(uint16_t);
        char buf[64+2];

        mrbc_object *obj = (mrbc_object *)mrbc_alloc(sizeof(mrbc_object));
        if (obj==NULL) {
            return LOAD_FILE_IREP_ERROR_ALLOCATION;
        }
        switch (tt) {
        case 0: { 	// IREP_TT_STRING
            obj->tt  = MRBC_TT_STRING;
            obj->sym = (char *)p;
        } break;
        case 1: { 	// IREP_TT_FIXNUM
            MEMCPY((uint8_t *)buf, p, obj_size);
            buf[obj_size] = '\0';
            
            obj->tt = MRBC_TT_FIXNUM;
            obj->i = (int)ATOI(buf);
        } break;
#if MRBC_USE_FLOAT
        case 2: { 	// IREP_TT_FLOAT
            MEMCPY((uint8_t *)buf, p, obj_size);
            buf[obj_size] = '\0';
            obj->tt = MRBC_TT_FLOAT;
            obj->f  = ATOF(buf);
        } break;
#endif
        default:
        	obj->tt = MRBC_TT_EMPTY;
        	break;
        }
        irep->pool[i] = obj;
        p += obj_size;
    }

    // SYMS BLOCK
    irep->sym = (uint8_t*)p;
    int sym_cnt =
    		irep->slen = _bin_to_uint32(p);	p += sizeof(uint32_t);
    while (--sym_cnt >= 0) {
        int len = _bin_to_uint16(p);		p += sizeof(uint16_t)+len+1;    // symbol_len+'\0'
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
_load_irep_0(const uint8_t **pos)
{
    // new irep
    mrbc_irep *irep = (mrbc_irep *)mrbc_alloc(sizeof(mrbc_irep));
    if (irep==NULL) {
        return NULL;
    }
    int ret = _load_irep_1(irep, pos);
    if (ret != NO_ERROR) {
    	mrbc_free(irep);
    	return NULL;
    }
    // recursively create the irep linked-list
    for (int i=0; i<irep->rlen; i++) {
        irep->irep_list[i] = _load_irep_0(pos);
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
_load_irep(mrbc_vm *vm, const uint8_t **pos)
{
    const uint8_t *p = *pos + 4;						// 4 = skip "IREP"
    int   sec_size = _bin_to_uint32(p); p += sizeof(uint32_t);

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
_load_lvar(mrbc_vm *vm, const uint8_t **pos)
{
    const uint8_t *p = *pos;

    /* size */
    *pos += _bin_to_uint32(p+sizeof(uint32_t));

    return NO_ERROR;
}

//================================================================
/*!@brief
  Load the VM bytecode.

  @param  vm    Pointer to VM.
  @param  ptr	Pointer to bytecode.

*/
__global__ void
guru_parse_bytecode(mrbc_vm *vm, const uint8_t *ptr)
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



