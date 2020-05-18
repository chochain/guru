/*! @file
  @brief
  GURU instruction unit
    1. guru VM, host or cuda image, constructor and dispatcher
    2. dumpers for regfile and irep tree

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  initialize VM
  	  allocate vm cuda memory
  	  parse mruby bytecodee
  	  dump irep tree (optional)
  execute VM
  	  opcode execution loop (on GPU)
  	    ucode_prefetch
  	    ucode_exec
  	    flush output per step (optional)
  </pre>
*/
#include <pthread.h>

#include "guru.h"
#include "base.h"
#include "util.h"
#include "mmu.h"
#include "symbol.h"
#include "c_string.h"

#include "class.h"
#include "state.h"
#include "vm.h"
#include "vmx.h"
#include "load.h"

#include "ucode.h"

__GURU__ void
VM::init(int id_, int step_)
{
	id    = id_;
	step  = step_;
	depth = err  = 0;
	run   = VM_STATUS_FREE;					// VM not allocated
}

//================================================================
// Fetch a VM for operation
// Note: thread 0 is the master controller, no other thread can
//       modify the VM status
//
#if GURU_HOST_GRIT_IMAGE
__GURU__ void
VM::prep(U8 *u8_gr)
{
	if (run==VM_STATUS_FREE) {
		_transcode(u8_gr);
		_ready(MEMOFF(U8PADD(u8_gr, ((GRIT*)u8_gr)->reps)));
    }
}
#else
__GURU__ void
VM::prep(U8 *ibuf)
{
	if (vm->run==VM_STATUS_FREE) {
		GRIT *gr = (GRIT*)parse_bytecode(ibuf);
		__transcode(gr);
		__ready(vm, gr);
    }
}
#endif // GURU_HOST_GRIT_IMAGE

//================================================================
/*!@brief
  execute one ISEQ instruction for each VM

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
__GURU__ void
VM::exec()
{
	static Ucode *uc_pool[MIN_VM_COUNT] = { NULL, NULL };
	if (!uc_pool[id]) {										// lazy allocation
		uc_pool[id] = new Ucode(this);
	}
	uc_pool[id]->run();										// whether my VM is completed
}

//================================================================
// Transcode Pooled objects and Symbol table recursively
// from source memory pointers to GR[] (i.e. regfile)
//
__GURU__ void
VM::_transcode(U8 *u8_gr)
{
	GRIT *gr = (GRIT*)u8_gr;
	GR   *r  = (GR*)U8PADD(gr, gr->pool);
	for (U32 i=0; i < gr->psz; i++, r++) {			// symbol table
		switch (r->gt) {
		case GT_SYM: guru_sym_rom(r);	break;
		case GT_STR: guru_str_rom(r);	break;		// instantiate the string
		default:
			// do nothing
			break;
        }
    }
}

//================================================================
/*!@brief
  VM initializer.

  @param  vm  Pointer to VM
 */
__GURU__ void
VM::_ready(GP irep)
{
	GR *r = regfile;
	for (U32 i=0; i<MAX_REGFILE_SIZE; i++, r++) {	// wipe register
		r->gt  = (i==0) ? GT_CLASS : GT_EMPTY;		// reg[0] is "self"
		r->acl = 0;
		r->off = (i==0) ? guru_rom_get_class(GT_OBJ) : 0;
	}
	state = NULL;
	run   = VM_STATUS_READY;
	depth = err = 0;

	StateMgr *sm = new StateMgr(this);				// needs a helper
	sm->push_state(irep, 0, regfile, 0);
}
