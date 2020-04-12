/*! @file
  @brief
  GURU bytecode loader (host load IREP code, build image and copy into CUDA memory).

  alternatively, load_gpu.cu can be used for device image building
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
#include "mmu.h"
#include "state.h"
#include "errorcode.h"

#include "load.h"

#if GURU_HOST_IMAGE
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
__HOST__ U32
bin_to_u32(const void *s)
{
    U32 x = *((U32*)s);
    return (x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24);
}

//================================================================
/*!@brief
  Get 16bit value from memory big endian.

  @param  s	Pointer of memory.
  @return	16bit unsigned value.
*/
__HOST__ U16
bin_to_u16(const void *s)
{
    U16 x = *((U16 *)s);
    return (x << 8) | (x >> 8);
}

__HOST__ int
_check_header(U8 **pos)
{
    const U8 *p = *pos;

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
    *pos += 22;

    return NO_ERROR;
}

//
// building memory image, offset-based with alignment
//
__HOST__ void
_to_gv(GV v[], U32 n, U8 *p, bool sym)
{
    // build POOL or SYM block
    char buf[64+1];
    for (U32 i=0; i < n; i++, v++) {
        U32  tt = sym ? 3 : *p++;
        U32  len = bin_to_u16(p);	p += sizeof(U16);

        switch (tt) {
        case 0:	// String
        	v->raw = p;
        	v->gt  = GT_STR;
        	break;
        case 1: // Integer (31-bit)
            memcpy(buf, p, len);
            buf[len] = '\0';
            v->i   = atoi(buf);
            v->gt  = GT_INT;
            break;
        case 2: // Float (32-bit)
            memcpy(buf, p, len);
            buf[len] = '\0';
            v->f   = (float)atof(buf);		// atof() returns double
            v->gt  = GT_FLOAT;
            break;
        case 3: // Symbol
        	v->raw = p;
        	v->gt  = GT_SYM;
        	break;
        default: // Others (not yet supported)
        	v->gt   = GT_NIL;
        	v->self = NULL;
        	break;
        }
        p += len + (sym ? 1 : 0);
    }
}

//================================================================
/*!@brief
  read one irep section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return       Pointer of allocated IREP or NULL

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
__HOST__ guru_irep*
_build_image(U8 **src)							// pos will be advance to next IREP block
{
	guru_irep irep;
    U8  *p = *src;

    // Header: sz, nlocals, nregs, rlen
    irep.size 	= bin_to_u32(p); 		p += sizeof(U32);			// IREP size
    irep.nv 	= bin_to_u16(p);		p += sizeof(U16);			// number of local variables
    irep.nr 	= bin_to_u16(p);		p += sizeof(U16);			// number of registers used
    irep.r  	= bin_to_u16(p);		p += sizeof(U16);			// number of child IREP blocks

    // ISEQ block
    irep.i 		= bin_to_u32(p);		p += sizeof(U32);			// ISEQ (bytecodes) length
    U8 *iseq    = (p += -(U32A)p & 3);								// ISEQ block (32-bit aligned)
    U32 iseq_sz = sizeof(U32)*irep.i;	p += iseq_sz;				// skip ISEQ (code) block
    U32 reps_sz = sizeof(guru_irep *) * irep.r;						// child REPS block
    U32 img_sz  = sizeof(guru_irep) + iseq_sz + reps_sz;
    guru_irep *tgt = (guru_irep *)cuda_malloc(ALIGN64(img_sz), 1);	// target CUDA IREP image (managed mem)
    assert(tgt);

#if GURU_DEBUG
    memset(tgt, 0xaa, img_sz);
#endif // GURU_DEBUG

    memcpy(tgt, &irep, sizeof(guru_irep));							// dup IREP header fields
    memcpy(U8PADD(tgt, sizeof(guru_irep)), iseq,  iseq_sz);			// copy ISEQ block
    tgt->size = img_sz;
    tgt->reps = (RIrep **)U8PADD(tgt, sizeof(guru_irep)+ALIGN(iseq_sz)); // pointer to child REPS

    // POOL block
    tgt->p   = bin_to_u32(p);			p += sizeof(U32);			// pool element count
    U8 *pool = p;
    for (U32 i=0; i<tgt->p; i++) {									// 1st pass (skim through pool)
    	U32 len = bin_to_u16(++p);		p += sizeof(U16)+len;
    }
    // SYM block
    tgt->s = bin_to_u32(p);				p += sizeof(U32);			// symbol element count
    U8 *sym = p;
    for (U32 i=0; i<tgt->s; i++) {									// 1st pass (skim through sym)
    	U32 len = bin_to_u16(p)+1;		p += sizeof(U16)+len;
    }
    *src = p;														// return source pointer

    // prep Register File block which combines Reps, Pooled objects & Symbol table
    U32 pool_sz = sizeof(GV) * (tgt->p + tgt->s);
    U8 *blk = (img_sz + pool_sz < CUDA_MIN_MEMBLOCK_SIZE)			// CUDA alloc 0x200B min
    	? U8PADD(tgt, img_sz)										// utilize free space if any
    	: (U8*)cuda_malloc(pool_sz, 1);
    assert(blk);

#if GURU_DEBUG
    memset(blk, 0xaa, pool_sz);
#endif // GURU_DEBUG

    tgt->pool = (GV *)blk;
    _to_gv(tgt->pool, 			tgt->s, sym,  1);					// symbol table 1st  (faster)
    _to_gv(tgt->pool + tgt->s,  tgt->p, pool, 0);					// pooled object 2nd (one extra calc)

    return tgt;														// position pointer ends here
}
//================================================================
/*!@brief
  Parse IREP section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return       Pointer of allocated IREP or NULL

  <pre>
  Structure
  "IREP"	section identifier
  0000_0000	section size
  "0000"	rite version
  </pre>
*/
__HOST__ guru_irep*
_load_irep(U8 **src)
{
	guru_irep *irep = _build_image(src);			// build CUDA image (in managed memory) from host image

    // recursively create the child irep tree
    for (U32 i=0; i < irep->r; i++) {				// number of irep children
    	irep->reps[i] = _load_irep(src);			// load a child irep recursively (from host image)
    }
    return irep;		// a pointer to CUDA irep (in managed memory)
}

//================================================================
/*!@brief
  Parse LVAR section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of LVAR section.
  @return int	zero if no error.
*/
__HOST__ U32
_load_lvar(U8 **pos)
{
    U8  *p     = *pos;
    U32 sec_sz = bin_to_u32(p+sizeof(U32));

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
__HOST__ U8 *
guru_parse_bytecode(U8 *src)
{
	U8  **sp = (U8 **)&src;			// a pointer to pointer, so that we can pass and adjust the pointer
	int ret  = _check_header(sp);

	U8 *irep;
    while (ret==NO_ERROR) {
        if (memcmp(*sp, "IREP", 4)==0) {
        	*sp += 4 + sizeof(U32);								// skip "IREP", irep_sz
            if (memcmp(*sp, "0000", 4) != 0) break;				// IREP version
            *sp += 4;											// skip "0000"

        	ret = ((irep = (U8*)_load_irep(sp))==NULL)
        			? LOAD_FILE_IREP_ERROR_ALLOCATION
        			: NO_ERROR;
        }
        else if (memcmp(*sp, "LVAR", 4)==0) {
            ret = _load_lvar(sp);
        }
        else if (memcmp(*sp, "END\0", 4)==0) {
            break;
        }
    }
    return (ret==NO_ERROR) ? irep : NULL;
}
#endif 	// GURU_HOST_IMAGE
