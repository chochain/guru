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

#define BU16(b)		(bin_to_u16((const void*)(b)))
#define BU32(b)		(bin_to_u32((const void*)(b)))

#if !GURU_HOST_IMAGE
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
_check_header(U8 **bp)
{
    const U8 *p = *bp;

    if (MEMCMP(p, "RITE000", 7)==0) {
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
    if (MEMCMP(p + 14, "MATZ", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_MATZ;
    }
    // Rite VM version
    // 0000: mruby 1.x
    // 0002: mruby 2.x
    if (MEMCMP(p + 18, "0000", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_VERSION;
    }

    *bp += 22;		// advance beyond header to IREP block

    return NO_ERROR;
}

__GURU__ void
_to_gv(GV v[], U32 n, U8 *p, bool sym)
{
    // build POOL or SYM block
    char buf[64+1];
    for (U32 i=0; i<n; i++, v++) {
        U32  tt = sym ? 3 : *p++;
        U32  len = BU16(p);	p += sizeof(U16);

        switch (tt) {
        case 0:	// String
        	v->raw = p;
        	v->gt  = GT_STR;
        	break;
        case 1: // Integer (31-bit)
            MEMCPY(buf, p, len);
            buf[len] = '\0';
            v->i   = ATOI(buf, 10);
            v->gt  = GT_INT;
            break;
        case 2: // Float (32-bit)
            MEMCPY(buf, p, len);
            buf[len] = '\0';
            v->f   = (float)ATOF(buf);		// atof() returns double
            v->gt  = GT_FLOAT;
            break;
        case 3: // Symbol
        	v->raw = p;						// TODO: need to move symbol to guru memory
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
__GURU__ guru_irep*
_build_image(U8 **bp)
{
    U8 *p = *bp;
    guru_irep r0;

    // sz,nlocals,nregs,rlen
    r0.size = BU32(p);	p += sizeof(U32);				// IREP block size
    r0.nv   = BU16(p);	p += sizeof(U16);				// number of local variables
    r0.nr   = BU16(p);	p += sizeof(U16);				// number of registers used
    r0.r    = BU16(p);	p += sizeof(U16);				// number of child IREP blocks

    // ISEQ block
    r0.i    = BU32(p);	p += sizeof(U32);				// ISEQ (bytecodes) length

    U8 *iseq    = (p += -(U32A)p & 3);					// 32-bit align code pointer
    U32 iseq_sz = sizeof(U32) * r0.i;	p += iseq_sz;	// skip ISEQ (code) block
    U32 reps_sz = sizeof(guru_irep) * r0.r;

    // POOL block
    r0.p = BU32(p);		p += sizeof(U32);				// pool element count
    U8 *pool = p;
    for (U32 i=0; i<r0.p; i++) {						// 1st pass (skim through pool)
    	U32 len = BU16(++p);	p += sizeof(U16)+len;
    }
    // SYM block
    r0.s = BU32(p);		p += sizeof(U32);				// symbol element count
    U8 *sym = p;
    for (U32 i=0; i<r0.s; i++) {						// 1st pass (skim through sym)
    	U32 len = BU16(p)+1;	p += sizeof(U16)+len;
    }
    *bp = p;

    // prep Register File block which combines Reps, Pooled objects & Symbol table
    U32 code_sz = sizeof(guru_irep) + reps_sz + iseq_sz;
    guru_irep *irep = (guru_irep*)guru_alloc(code_sz);
    *irep = r0;											// hardcopy IREP
    MEMCPY(irep+1, iseq, iseq_sz);						// copy ISEQ
    guru_irep **reps = (guru_irep**)U8PADD(irep, sizeof(guru_irep) + iseq_sz);
    irep->reps = reps;

    U32 pool_sz = sizeof(GV) * (r0.p + r0.s);
    irep->pool  = (GV *)guru_alloc(pool_sz);			// TODO: use vm->regfile[] directly

    _to_gv(irep->pool, 		  r0.s, sym,  1);			// symbol table 1st  (faster)
    _to_gv(irep->pool + r0.s, r0.p, pool, 0);			// pooled object 2nd (one extra calc)

    return irep;
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
    guru_irep *irep = _build_image(bp);

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
    int ret = _check_header(bp);

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
