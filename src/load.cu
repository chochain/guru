/*! @file
  @brief
  Guru bytecode loader.

  <pre>
  Copyright (C) 2015-2017 Kyushu Institute of Technology.
  Copyright (C) 2015-2017 Shimane IT Open-Innovation Center.

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
#if GURU_HOST_IMAGE

__HOST__ uint32_t
bin_to_uint32(const void *s)
{
    uint32_t x = *((uint32_t *)s);
    return (x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24);
}

//================================================================
/*!@brief
  Get 16bit value from memory big endian.

  @param  s	Pointer of memory.
  @return	16bit unsigned value.
*/
__HOST__ uint16_t
bin_to_uint16(const void *s)
{
    uint16_t x = *((uint16_t *)s);
    return (x << 8) | (x >> 8);
}

__HOST__ int
_load_header(const uint8_t **pos)
{
    const uint8_t *p = *pos;

    if (memcmp(p, "RITE0004", 8) != 0) {
        return LOAD_FILE_HEADER_ERROR_VERSION;
    }
    /* Ignore CRC */
    /* Ignore size */

    if (memcmp(p + 14, "MATZ", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_MATZ;
    }
    if (memcmp(p + 18, "0000", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_VERSION;
    }
    *pos += 22;

    return NO_ERROR;
}

//
// building memory image
//
__HOST__ void
_build_image(uint8_t **pirep, uint8_t *src)
{
	guru_irep *irep = (guru_irep *)*pirep;
	uint8_t   *base = (uint8_t *)irep;

    int list_sz = sizeof(uint32_t) * (irep->rlen + (irep->rlen & 1));
    int iseq_sz = sizeof(uint32_t) * (irep->ilen + (irep->ilen & 1));
    int pool_sz = sizeof(uint32_t) * (irep->plen + (irep->plen & 1));
    int sym_sz  = sizeof(uint32_t) * (irep->slen + (irep->slen & 1));

	uint8_t *list = base + sizeof(guru_irep) + ((8 - sizeof(guru_irep)) & 7);
	uint8_t *iseq = list + list_sz;
	uint8_t *pool = iseq + iseq_sz;
	uint8_t *sym  = pool + pool_sz;
	uint8_t *tstr = sym  + sym_sz;				// extended string table

    memcpy(iseq, src + irep->iseq, iseq_sz);	// copy ISEQ block
    irep->iseq = (uint32_t)(iseq - base);

    uint8_t *p = src + irep->pool;				// build object pool
    for (int i = 0; i < irep->plen; i++) {
        int  tt = *p++;
        int  len = bin_to_uint16(p);	p += sizeof(uint16_t);
        int  asz = len + ((8 - len) & 7);
        char buf[64+2];

        uint32_t *v = (uint32_t *)(pool + i * sizeof(uint32_t));
        switch (tt) {
        case 0: { 								// IREP_TT_STRING
            *v = (uint32_t)(tstr - base);
            memcpy(tstr, p, asz);				// string on 4-byte boundary
            tstr += asz;
        } break;
        case 1: { 								// IREP_TT_FIXNUM
            memcpy((uint8_t *)buf, p, len);
            buf[len] = '\0';
            *v = atoi(buf)<<1;					// mark as number i.e. LSB=0
        } break;
#if GURU_USE_FLOAT
        case 2: { 								// IREP_TT_FLOAT
            memcpy((uint8_t *)buf, p, len);
            buf[len] = '\0';
            *(mrbc_float *)v = atof(buf);
            *v |= 1;							// mark as float  i.e. LSB=1
        } break;
#endif
        default:
        	*v = 0;		// other object are not supported yet
        	break;
        }
        p += len;
    }
    irep->pool = (uint32_t)(pool - base);

    p = src + irep->sym;						// build symbol table
    for (int i=0; i < irep->slen; i++) {
        int len = bin_to_uint16(p)+1;		p += sizeof(uint16_t);	// symbol_len+'\0'
        int asz = len + ((8 - len) & 7);		// 8-byte alignment

        uint32_t *v = (uint32_t *)(sym + i * sizeof(uint32_t));
        *v = (uint32_t)(tstr - base);

        memcpy(tstr, p, asz);					// copy raw string to string table
        tstr += asz;
        p    += len;
    }
    irep->sym  = (uint32_t)(sym - base);
    irep->size = (uint32_t)(tstr - base);

    *pirep = tstr;								// return the adjusted irep pointer
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
__HOST__ void
_get_irep_size(guru_irep *irep, const uint8_t **pos)
{
    const uint8_t *p = *pos + 4;			// skip "IREP"

    // nlocals,nregs,rlen
    irep->nlv  = bin_to_uint16(p);	p += sizeof(uint16_t);		// number of local variables
    irep->nreg = bin_to_uint16(p);	p += sizeof(uint16_t);		// number of registers used
    irep->rlen = bin_to_uint16(p);	p += sizeof(uint16_t);		// number of child IREP blocks
    irep->ilen = bin_to_uint32(p);	p += sizeof(uint32_t);		// ISEQ (bytecodes) length

    p += ((uint8_t *)irep - p) & 0x03;							// 32-bit align code pointer

    irep->iseq = (uint32_t)(p - *pos);	p += irep->ilen * sizeof(uint32_t);	// ISEQ (code) block
    irep->plen = bin_to_uint32(p);		p += sizeof(uint32_t);				// POOL block

    irep->pool = (uint32_t)(p - *pos);										// pool offset
    irep->list = sizeof(guru_irep) + ((8 - sizeof(guru_irep)) & 7);			// irep list offset

    for (int i = 0; i < irep->plen; i++) {	// scan through pool (so we know the size to allocate)
        int  tt  = *p++;
        int  len = bin_to_uint16(p);    p += sizeof(uint16_t) + len;
    }

    irep->slen = bin_to_uint32(p);		p += sizeof(uint32_t);
    irep->sym  = (uint32_t)(p - *pos);										// symbol offset
    for (int i=0; i < irep->slen; i++) {
        int len = bin_to_uint16(p)+1;	p += sizeof(uint16_t) + len;	   	// symbol_len+'\0'
    }
    *pos = p;								// position pointer ends here
}
//================================================================
/*!@brief
  read all irep section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return       Pointer of allocated mrbc_irep or NULL
*/
__HOST__ guru_irep*
_load_irep_0(uint8_t **pirep, const uint8_t **pos)
{
	guru_irep *irep = (guru_irep *)*pirep;
	uint8_t   *src  = (uint8_t *)*pos;
	uint8_t   *base = (uint8_t *)irep;

	_get_irep_size(irep, pos);			// populate metadata of current IREP
    _build_image(pirep, src);

    // recursively create the child irep tree
    uint32_t *p = (uint32_t *)(base + irep->list);
    for (int i = 0; i < irep->rlen; i++) {
    	guru_irep *irep_n = _load_irep_0(pirep, pos);	// a child irep
        p[i] = (uint32_t)((uint8_t *)irep_n - base);	// calculate offset
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
__HOST__ guru_irep*
_load_irep(const uint8_t **pos)
{
    const uint8_t *p = *pos + 4;					// 4 = skip "IREP"
    int   sec_sz = bin_to_uint32(p); p += sizeof(uint32_t);

    if (memcmp(p, "0000", 4) != 0) return NULL;		// IREP version
    p += 4;											// 4 = skip "0000"

    int     irep_max_sz = sec_sz * 2;
    uint8_t *ibuf = (uint8_t *)guru_malloc(irep_max_sz, 1);
    uint8_t *ipos = ibuf;
    if (!ibuf) return NULL;							//

    memset(ibuf, 0xaa, irep_max_sz);

    guru_irep *irep = _load_irep_0(&ibuf, &p);		// recursively load irep tree

    if (irep==NULL) return NULL;					// allocation error

    int irep_sz = ibuf - ipos;
    assert(irep_sz <= irep_max_sz);					// verify allocated buffer size
    assert(sec_sz==(p - *pos));						// verify pointer arith.

    *pos += sec_sz;

    return irep;
}

//================================================================
/*!@brief
  Parse LVAR section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of LVAR section.
  @return int	zero if no error.
*/
__HOST__ int
_load_lvar(const uint8_t **pos)
{
    const uint8_t *p = *pos;

    int sec_sz = bin_to_uint32(p+sizeof(uint32_t));

    // TODO: local variable is not supported yet

    *pos += sec_sz;

    return NO_ERROR;
}

//================================================================
/*!@brief
  Load the VM bytecode.

  @param  vm    Pointer to VM.
  @param  ptr	Pointer to bytecode.

*/
__HOST__ void
guru_parse_bytecode(guru_vm *vm, const uint8_t *ptr)
{
    int ret = _load_header(&ptr);

    while (ret==NO_ERROR) {
        if (memcmp(ptr, "IREP", 4)==0) {
        	ret = (vm->irep = _load_irep(&ptr))==NULL
        			? LOAD_FILE_IREP_ERROR_ALLOCATION
        			: NO_ERROR;
        }
        else if (memcmp(ptr, "LVAR", 4)==0) {
            ret = _load_lvar(&ptr);
        }
        else if (memcmp(ptr, "END\0", 4)==0) {
            break;
        }
    }
}
#ifdef GURU_DEBUG
__HOST__ void
_show_irep(guru_irep *irep, char level, uint32_t idx)
{
	static char n = 'a';
	printf("\tirep[%c]=%c%04x: size=%d, nreg=%d, nlocal=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			n++, level, idx,
			irep->size, irep->nreg, irep->nlv, irep->plen, irep->slen, irep->rlen, irep->ilen);

	// dump all children ireps
	uint8_t  *base = (uint8_t *)irep;
	uint32_t *off  = (uint32_t *)(base + irep->list);
	for (int i=0; i<irep->rlen; i++, off++) {
		_show_irep((guru_irep *)(base + *off), level+1, *off);
	}
}

__HOST__ void
guru_show_irep(guru_irep *irep)
{
	_show_irep(irep, 'A', 0);
}
#endif

#else	// !GURU_HOST_IMAGE

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
__GURU__ int
_load_irep_1(mrbc_irep *irep, const uint8_t **pos)
{
    const uint8_t *p = *pos + 4;		// skip "IREP"

    // nlocals,nregs,rlen
    irep->nlv  = _bin_to_uint16(p);	p += sizeof(uint16_t);		// number of local variables
    irep->nreg = _bin_to_uint16(p);	p += sizeof(uint16_t);		// number of registers used
    irep->rlen = _bin_to_uint16(p);	p += sizeof(uint16_t);		// number of child IREP blocks
    irep->ilen = _bin_to_uint32(p);	p += sizeof(uint32_t);		// ISEQ (bytecodes) length

    p += ((uint8_t *)irep - p) & 0x03;	// 32-bit align code pointer

    irep->iseq = (uint32_t *)p;			p += irep->ilen * sizeof(uint32_t);		// ISEQ (code) block
    irep->plen = _bin_to_uint32(p);		p += sizeof(uint32_t);					// POOL block

    if (irep->rlen) {					// allocate child irep's pointers (later filled by _load_irep_0)
        irep->list = (mrbc_irep **)mrbc_alloc(sizeof(mrbc_irep *) * irep->rlen);
        if (irep->list==NULL) {
            return LOAD_FILE_IREP_ERROR_ALLOCATION;
        }
    }
    if (irep->plen) {					// allocate pool of object pointers
        irep->pool = (mrbc_object**)mrbc_alloc(sizeof(void*) * irep->plen);
        if (irep->pool==NULL) {
            return LOAD_FILE_IREP_ERROR_ALLOCATION;
        }
    }

    for (int i = 0; i < irep->plen; i++) {		// build object pool
        int  tt = *p++;
        int  obj_size = _bin_to_uint16(p);	p += sizeof(uint16_t);
        char buf[64+2];

        mrbc_object *obj = (mrbc_object *)mrbc_alloc(sizeof(mrbc_object));
        if (obj==NULL) {
            return LOAD_FILE_IREP_ERROR_ALLOCATION;
        }
        switch (tt) {
        case 0: { 	// IREP_TT_STRING
            obj->tt  = GURU_TT_STRING;
            obj->sym = (char *)p;
        } break;
        case 1: { 	// IREP_TT_FIXNUM
            MEMCPY((uint8_t *)buf, p, obj_size);
            buf[obj_size] = '\0';

            obj->tt = GURU_TT_FIXNUM;
            obj->i = (int)ATOI(buf);
        } break;
#if GURU_USE_FLOAT
        case 2: { 	// IREP_TT_FLOAT
            MEMCPY((uint8_t *)buf, p, obj_size);
            buf[obj_size] = '\0';
            obj->tt = GURU_TT_FLOAT;
            obj->f  = ATOF(buf);
        } break;
#endif
        default:
        	obj->tt = GURU_TT_EMPTY;	// other object are not supported yet
        	break;
        }
        irep->pool[i] = obj;			// stick it into object pool array
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
    int ret = _load_irep_1(irep, pos);		// populate content of current IREP
    if (ret != NO_ERROR) {
    	mrbc_free(irep);
    	return NULL;
    }
    // recursively create the child irep tree
    for (int i=0; i<irep->rlen; i++) {
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
__GPU__ void
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

__HOST__ void
guru_show_irep(mrbc_irep *irep)
{
	printf("\tnregs=%d, nlocals=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			irep->nreg, irep->nlv, irep->plen, irep->slen, irep->rlen, irep->ilen);

	// dump all children ireps
	for (int i=0; i<irep->rlen; i++) {
		guru_show_irep(irep->list[i], level+1);
	}
}
#endif

