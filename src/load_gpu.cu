/*! @file
  @brief
  GURU bytecode loader (IREP code parse by CUDA device directly).

  alternatively, load.cu can be used for host built image (then passed into device for execution)
  <pre>
  Copyright (C) 2019- Greeni

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "guru.h"
#include "mmu.h"
#include "util.h"

#include "base.h"
#include "state.h"
#include "vm.h"
#include "load.h"
#include "errorcode.h"

#define BU16(b)		(bin_to_u16((const void*)b))
#define BU32(b)		(bin_to_u32((const void*)b))

#if !GURU_HOST_IMAGE
__HOST__ int
_check_header(U8 **bp)
{
    const U8 *p = *bp;

    if (memcmp(p, "RITE000", 7)==0) {
    	// Rite binary version
    	// 0002: mruby 1.0
    	// 0003: mruby 1.1, 1.2
    	// 0004: mruby 1.3, 1.4
    	// 0005: mruby 2.0
    	U8 c = *(p+7);
        if (c < '3' || c > '4') {
        	return LOAD_FILE_HEADER_ERROR_VERSION;
        }
    }
    /* Ignore CRC */
    /* Ignore size */
    if (memcmp(p + 14, "MATZ", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_MATZ;
    }
    // Rite VM version
    // 0000: mruby 1.x
    // 0002: mruby 2.x
    if (memcmp(p + 18, "0000", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_VERSION;
    }
    *bp += 22;

    return NO_ERROR;
}

//================================================================
/*!@brief
  Parse header section.

  @param  bp	A pointer of pointer of RITE header.
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
_load_header(U8 **bp)
{
    U8 *p = *bp;

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
    *bp += 22;

    return NO_ERROR;
}

//================================================================
/*!@brief
  read one irep section.

  @param  bp	A pointer of pointer of IREP section.
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
_build_image(guru_irep *irep, U8 **bp)
{
    U8 *p = *bp;

    // nlocals,nregs,rlen
    irep->nv = BU16(p);	p += sizeof(U16);		// number of local variables
    irep->nr = BU16(p);	p += sizeof(U16);		// number of registers used
    irep->r  = BU16(p);	p += sizeof(U16);		// number of child IREP blocks
    irep->i  = BU32(p);	p += sizeof(U32);		// ISEQ (bytecodes) length

    p += U8POFF(irep, p) & 0x03;						// 32-bit align code pointer
    p += irep->i * sizeof(U32);							// skip ISEQ (code) block
    irep->p  = BU32(p);	p += sizeof(U32);		// POOL block

    if (irep->r) {			// allocate child irep's pointers (later filled by _load_irep_0)
        irep->reps = (guru_irep **)guru_alloc(sizeof(guru_irep *) * irep->r);
    }
    if (irep->p) {			// allocate pool of object pointers
        irep->pool = (GV*)guru_alloc(sizeof(GV) * irep->p);
    }

    for (U32 i=0; i < irep->p; i++) {		// build object pool
        U32 tt = *p++;
        U32 obj_size = BU16(p);	p += sizeof(U16);
        U8  buf[64+2];

        GV *obj = (GV*)guru_alloc(sizeof(GV));
        switch (tt) {
        case 0: { 	// IREP_GT_STRING
            obj->gt  = GT_STR;
            obj->raw = p;
        } break;
        case 1: { 	// IREP_GT_FIXNUM
            MEMCPY(buf, p, obj_size);
            buf[obj_size] = '\0';

            obj->gt = GT_INT;
            obj->i = (int)ATOI(buf, 10);
        } break;
        case 2: { 	// IREP_GT_FLOAT
            MEMCPY(buf, p, obj_size);
            buf[obj_size] = '\0';
            obj->gt = GT_FLOAT;
            obj->f  = ATOF(buf);
        } break;
        default:
        	obj->gt = GT_EMPTY;	// other object are not supported yet
        	break;
        }
        irep->pool[i] = *obj;			// stick it into object pool array
        p += obj_size;
    }

    // SYMS BLOCK
    irep->s = BU32(p);			p += sizeof(U32);
    for (U32 i=0; i<irep->s; i++) {
        int len = BU16(p);		p += sizeof(U16)+len+1;    // symbol_len+'\0'
    }
    *bp = p;

    return NO_ERROR;
}

//================================================================
/*!@brief
  read all irep section.

  @param  bp	A pointer of pointer of IREP section.
  @return       Pointer of allocated mrbc_irep or NULL

  <pre>
  Structure
  "IREP"	section identifier
  0000_0000	section size
  "0000"	rite version
  </pre>
*/
__GURU__ guru_irep*
_load_irep(U8 **bp)
{
    // allocate new irep
    guru_irep *irep = (guru_irep *)guru_alloc(sizeof(guru_irep));

    int ret = _build_image(irep, bp);		// populate content of current IREP
    if (ret != NO_ERROR) {
    	guru_free(irep);
    	return NULL;
    }
    // recursively create the child irep tree
    for (U32 i=0; i<irep->r; i++) {
        irep->reps[i] = _load_irep(bp);
    }
    return irep;
}

//================================================================
/*!@brief
  Parse LVAR section.

  @param  bp	A pointer of pointer of LVAR section.
  @return int	zero if no error.
*/
__GURU__ int
_load_lvar(U8 **bp)
{
    U8 *p = *bp;

    /* size */
    *bp += BU32(p+sizeof(U32));

    return NO_ERROR;
}

//================================================================
/*!@brief
  Load the VM bytecode.

  @param  src	Pointer to bytecode.
*/
__GURU__ U8*
parse_bytecode(U8 *src)
{
	U8 **bp = &src;
    int ret = _load_header(bp);

    U8 *irep;
    while (ret==NO_ERROR) {
        if (MEMCMP(*bp, "IREP", 4)==0) {
        	*bp += 4 + sizeof(U32);								// skip "IREP", irep_sz
            if (MEMCMP(*bp, "0000", 4) != 0) break;				// IREP version
            *bp += 4;											// skip "0000"

        	ret = ((irep=(U8*)_load_irep(bp))==NULL)
                	? LOAD_FILE_IREP_ERROR_ALLOCATION
        			: NO_ERROR;
        }
        else if (MEMCMP(*bp, "LVAR", 4)==0) {
            ret = _load_lvar(bp);
        }
        else if (MEMCMP(*bp, "END\0", 4)==0) {
            break;
        }
    }
    return (ret==NO_ERROR) ? irep : NULL;
}
#endif // !GURU_HOST_IMAGE
