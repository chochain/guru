/*! @file
  @brief
  guru microcode unit

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  1. VM attribute accessor macros
  2. internal state management functions
  3. a list of opcode (microcode) executor, and
  4. the core opcode dispatcher
  </pre>
*/

#include "alloc.h"
#include "store.h"
#include "static.h"
#include "symbol.h"
#include "global.h"

#include "console.h"

#include "opcode.h"
#include "object.h"
#include "class.h"

#if GURU_USE_STRING
#include "c_string.h"
#endif
#if GURU_USE_ARRAY
#include "c_range.h"
#include "c_array.h"
#include "c_hash.h"
#endif

__GURU__ U32 _mutex_op;

//
// becareful with the following macros, because they release regs[ra] first
// so, make sure value is kept before the release
//
#define _RA_X(r)    do { mrbc_release(&regs[ra]); regs[ra] = *(r); mrbc_retain(r); } while (0)
#define _RA_V(v)    do { mrbc_release(&regs[ra]); regs[ra] = (v); } while (0)
#define _RA_T(t, e) do { mrbc_release(&regs[ra]); regs[ra].tt = (t); regs[ra].e; } while (0)

#if GURU_HOST_IMAGE
//================================================================
/*! get sym[n] from symbol table in irep

  @param  p	Pointer to IREP SYMS section.
  @param  n	n th
  @return	symbol name string
*/
__GURU__ U8P
_vm_symbol(guru_vm *vm, U32 n)
{
	guru_irep *irep = VM_IREP(vm);
	U32P p = (U32P)U8PADD(irep, irep->sym  + n * sizeof(U32));

	return U8PADD(irep, *p);
}

__GURU__ mrbc_sym
_vm_symid(guru_vm *vm, U32 n)
{
	const U8P name = _vm_symbol(vm, n);
	return name2symid(name);
}

__GURU__ U32P
_vm_pool(guru_vm *vm, U32 n)
{
	guru_irep *irep = VM_IREP(vm);

	return (U32P)U8PADD(irep, irep->pool + n*sizeof(U32));
}

__GURU__ guru_irep*
_vm_irep_list(guru_vm *vm, U32 n)
{
	guru_irep *irep = VM_IREP(vm);
	U32P p = (U32P)U8PADD(irep, irep->list + n*sizeof(U32));

	return (guru_irep *)U8PADD(irep, *p);
}

#else  // !GURU_HOST_IMAGE

__GURU__ U8P
_get_symbol(const U8P p, U32 n)
{
    U32 max = _bin_to_u32(p);		p += sizeof(U32);
    if (n >= max) return NULL;

    for (; n>0; n--) {	// advance to n'th symbol
        U16 s = _bin_to_u16(p);		p += sizeof(U16)+s+1;	// symbol len + '\0'
    }
    return (U8P)p + sizeof(U16);  	// skip size(2 bytes)
}

__GURU__ mrbc_sym
_get_symid(const U8P p, U32 n)
{
  	const U8P name = _get_symbol(p, n);
    return name2symid(name);
}
#endif

//================================================================
/*!@brief
  Push current status to callinfo stack

*/
__GURU__ void
_push_state(guru_vm *vm, U32 argc)
{
	guru_state *top = vm->state;
    guru_state *st  = (guru_state *)mrbc_alloc(sizeof(guru_state));

    st->reg   = top->reg;			// pass register file
    st->irep  = top->irep;
    st->pc 	  = top->pc;
    st->argc  = argc;				// allocate local stack
    st->klass = top->klass;
    st->prev  = top;

    vm->state = st;
}

//================================================================
/*!@brief
  Push current status to callinfo stack

*/
__GURU__ void
_pop_state(guru_vm *vm, mrbc_value *regs)
{
    guru_state *st = vm->state;
    
    vm->state = st->prev;
    
    mrbc_value *p  = regs+1;			// clear stacked arguments
    for (U32 i=0; i < st->argc; i++) {
        mrbc_release(p++);
    }
    mrbc_free(st);
}

__GURU__ void
_vm_proc_call(guru_vm *vm, mrbc_value v[], U32 argc)
{
	_push_state(vm, argc);				// check _funcall which is not used

	vm->state->pc   = 0;
	vm->state->irep = v[0].proc->irep;	// switch into callee context
	vm->state->reg  = v;				// shift register file pointer (for local stack)

	v[0].proc->refc++;					// CC: 20181027 added to track proc usage
}

// Object.new
__GURU__ void
_vm_object_new(guru_vm *vm, mrbc_value v[], U32 argc)
{
    mrbc_value obj = mrbc_store_new(v[0].cls, 0);
    //
    // build a temp IREP for calling "initialize"
    // TODO: make the image static
    //
    const U32 iseq = sizeof(guru_irep);				// iseq   offset
    const U32 sym  = iseq + 2 * sizeof(U32);		// symbol offset
    const U32 stbl = iseq + 4 * sizeof(U32);		// symbol string table
    guru_irep irep[2] = {
        {
            0,                  		// size (u32)
            0, 0, 0, 2, 0, 0,   		// nlv, rreg, rlen, ilen, plen, slen (u16)
            iseq, sym, 0, 0   			// iseq (u32), sym (u32), pool, list
        },
        {
        	(MKOPCODE(OP_SEND)|MKARG_A(0)|MKARG_B(0)|MKARG_C(argc)),	// ISEQ block
        	(MKOPCODE(OP_ABORT)) & 0xffff, (MKOPCODE(OP_ABORT)) >> 16,
        	stbl & 0xffff, stbl >> 16, 	0xaaaa, 0xaaaa,					// symbol table
        	0x74696e69, 0x696c6169, 0x0a00657a, 0xaaaaaaaa				// "initialize"
        }
    };

    mrbc_release(&v[0]);
    v[0] = obj;
    mrbc_retain(&obj);

    // context switch, which is not multi-thread ready
    // TODO: create a vm context object with separate regfile
    guru_state  *st    = vm->state;

    uint16_t    pc0    = st->pc;
    mrbc_value  *reg0  = st->reg;
    guru_irep 	*irep0 = st->irep;

    st->pc 	 = 0;
    st->irep = irep;
    st->reg  = v;		   // new register file (shift for call stack)

    while (guru_op(vm)==0); // run til ABORT, or exception

    st->pc 	 = pc0;
    st->reg  = reg0;
    st->irep = irep0;

    SET_RETURN(obj);
}

//================================================================
/*!@brief
  Execute OP_NOP

  No operation

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_nop(guru_vm *vm, U32 code, mrbc_value *regs)
{
    return 0;
}

//================================================================
/*!@brief
  Execute OP_MOVE

  R(A) := R(B)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_move(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);

    _RA_X(&regs[rb]);                  // swap ra <= rb

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADL

  R(A) := Pool(Bx)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_loadl(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    U32P p = _vm_pool(vm, rb);
    mrbc_object obj;

    if (*p & 1) {
    	obj.tt = GURU_TT_FLOAT;
    	obj.f  = *(mrbc_float *)p;
    }
    else {
    	obj.tt = GURU_TT_FIXNUM;
    	obj.i  = *p>>1;
    }
    _RA_V(obj);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADI

  R(A) := sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_loadi(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_sBx(code);

    _RA_T(GURU_TT_FIXNUM, i=rb);

    return 0;
}


//================================================================
/*!@brief
  Execute OP_LOADSYM

  R(A) := Syms(Bx)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_loadsym(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra  = GETARG_A(code);
    int rb  = GETARG_Bx(code);

    mrbc_sym sid = _vm_symid(vm, rb);

    _RA_T(GURU_TT_SYMBOL, i=sid);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADNIL

  R(A) := nil

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_loadnil(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    _RA_T(GURU_TT_NIL, i=0);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADSELF

  R(A) := self

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_loadself(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    _RA_X(&regs[0]);                   // ra <= vm

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADT

  R(A) := true

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_loadt(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    _RA_T(GURU_TT_TRUE, i=0);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADF

  R(A) := false

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_loadf(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    _RA_T(GURU_TT_FALSE, i=0);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GETGLOBAL

  R(A) := getglobal(Syms(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_getglobal(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra  = GETARG_A(code);
    int rb  = GETARG_Bx(code);

    mrbc_sym sid = _vm_symid(vm, rb);
    mrbc_value v = global_object_get(sid);
    _RA_V(v);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_SETGLOBAL

  setglobal(Syms(Bx), R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_setglobal(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra  = GETARG_A(code);
    int rb  = GETARG_Bx(code);

    mrbc_sym sid = _vm_symid(vm, rb);
    global_object_add(sid, regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GETIV

  R(A) := ivget(Syms(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_getiv(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    const U8P name = _vm_symbol(vm, rb);
    mrbc_sym sid   = name2symid(name+1);		// skip '@'
    mrbc_value ret   = mrbc_store_get(&regs[0], sid);

    _RA_V(ret);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_SETIV

  ivset(Syms(Bx),R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_setiv(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    const U8P name = _vm_symbol(vm, rb);
    mrbc_sym  sid  = name2symid(name+1);	// skip '@'

    mrbc_store_set(&regs[0], sid, &regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GETCONST

  R(A) := constget(Syms(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_getconst(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra  = GETARG_A(code);
    int rb  = GETARG_Bx(code);

    mrbc_sym  sid = _vm_symid(vm, rb);
    mrbc_value v  = const_object_get(sid);

    _RA_X(&v);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_SETCONST

  constset(Syms(Bx),R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_setconst(guru_vm *vm, U32 code, mrbc_value *regs) {
    int ra  = GETARG_A(code);
    int rb  = GETARG_Bx(code);

    mrbc_sym sid = _vm_symid(vm, rb);
    const_object_add(sid, &regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GETUPVAR

  R(A) := uvget(B,C)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_getupvar(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);   		// UP

    guru_state *st = vm->state;

    int n = (rc+1) << 1;			// depth of call stack
    while (n > 0){					// walk up call stack
        st = st->prev;
        n--;
    }
    mrbc_value *uregs = st->reg;	// outer scope register file

    _RA_X(&uregs[rb]);             	// ra <= up[rb]

    return 0;
}

//================================================================
/*!@brief
  Execute OP_SETUPVAR

  uvset(B,C,R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_setupvar(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);   				// UP level

    guru_state *st = vm->state;

    int n = (rc+1) << 1;					// 2 per outer scope level
    while (n > 0){
        st = st->prev;
        n--;
    }
    mrbc_value *uregs = st->reg;

    mrbc_release(&uregs[rb]);
    uregs[rb] = regs[ra];                   // update outer-scope vars
    mrbc_retain(&regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_JMP

  pc += sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_jmp(guru_vm *vm, U32 code, mrbc_value *regs)
{
    vm->state->pc += GETARG_sBx(code) - 1;
    return 0;
}

//================================================================
/*!@brief
  Execute OP_JMPIF

  if R(A) pc += sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_jmpif (guru_vm *vm, U32 code, mrbc_value *regs)
{
    if (regs[GETARG_A(code)].tt > GURU_TT_FALSE) {
        vm->state->pc += GETARG_sBx(code) - 1;
    }
    return 0;
}

//================================================================
/*!@brief
  Execute OP_JMPNOT

  if not R(A) pc += sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_jmpnot(guru_vm *vm, U32 code, mrbc_value *regs)
{
    if (regs[GETARG_A(code)].tt <= GURU_TT_FALSE) {
        vm->state->pc += GETARG_sBx(code) - 1;
    }
    return 0;
}


//================================================================
/*!@brief
  Execute OP_SEND / OP_SENDB

  OP_SEND   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C))
  OP_SENDB  R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C),&R(A+C+1))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_send(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);  // index of method sym
    int rc = GETARG_C(code);  // number of params
    mrbc_value rcv = regs[ra];

    // Clear block param (needed ?)
    int bidx = ra + rc + 1;
    switch(GET_OPCODE(code)) {
    case OP_SEND:
        mrbc_release(&regs[bidx]);
        regs[bidx].tt = GURU_TT_NIL;
        break;
    case OP_SENDB:						// set Proc object
        if (regs[bidx].tt != GURU_TT_NIL && regs[bidx].tt != GURU_TT_PROC){
            // TODO: fix the following behavior
            // convert to Proc ?
            // raise exceprion in mruby/c ?
            return 0;
        }
        break;
    default: break;
    }

	mrbc_sym  sid = _vm_symid(vm, rb);
    mrbc_proc *m  = (mrbc_proc *)mrbc_get_proc_by_symid(rcv, sid);

    if (m==0) {
    	const U8P name = _vm_symbol(vm, rb);
    	console_na(name);							// dump error, bail out
    	return 0;
    }
    if (IS_CFUNC(m)) {					// is m->func a C function
        if (m->func==c_proc_call) {
        	_vm_proc_call(vm, regs+ra, rc);
        }
        else if (m->func==c_object_new) {
        	_vm_object_new(vm, regs+ra, rc);
        }
        else {
        	m->func(regs+ra, rc);					// call the C-func
        	for (U32 i=ra+1; i<=bidx; i++) {		// clean up block parameters
                mrbc_release(&regs[i]);
            }
        }
    }
    else {								// m->func is a Ruby function (aka IREP)
    	_push_state(vm, rc);			// append callinfo list

    	vm->state->irep = m->irep;		// call into target context
    	vm->state->pc 	= 0;			// call into target context
    	vm->state->reg 	+= ra;			// add call stack (new register)
    }
    return 0;
}

//================================================================
/*!@brief
  Execute OP_CALL

  R(A) := self.call(frame.argc, frame.argv)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_call(guru_vm *vm, U32 code, mrbc_value *regs)
{
    _push_state(vm, 0);

    // jump to proc
    vm->state->pc 	= 0;
    vm->state->irep = regs[0].proc->irep;

    return 0;
}



//================================================================
/*!@brief
  Execute OP_ENTER

  arg setup according to flags (23=5:5:1:5:5:1:1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_enter(guru_vm *vm, U32 code, mrbc_value *regs)
{
    U32 param = GETARG_Ax(code);

    U32 arg0 = (param >> 13) & 0x1f;  // default args
    U32 argc = (param >> 18) & 0x1f;  // given args

    if (arg0 > 0){
        vm->state->pc += vm->state->argc - argc;
    }
    return 0;
}

//================================================================
/*!@brief
  Execute OP_RETURN

  return R(A) (B=normal,in-block return/break)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_return(guru_vm *vm, U32 code, mrbc_value *regs)
{
    // return value
    int ra = GETARG_A(code);
    mrbc_value ret = regs[ra];

    mrbc_release(&regs[0]);
    regs[0]     = ret;
    regs[ra].tt = GURU_TT_EMPTY;

    _pop_state(vm, regs);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_BLKPUSH

  R(A) := block (16=6:1:5:4)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_blkpush(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    mrbc_value *stack = regs + 1;       // call stack: push 1 mrbc_value

    if (stack[0].tt==GURU_TT_NIL){		// Check leak?
        return vm->err = 255;  			// EYIELD
    }
    _RA_X(stack);                       // ra <= stack[0]

    return 0;
}

//================================================================
/*!@brief
  Execute OP_ADD

  R(A) := R(A)+R(A+1) (Syms[B]=:+,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_add(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    mrbc_value *r0 = &regs[ra];
    mrbc_value *r1 = &regs[ra+1];

    if (r0->tt==GURU_TT_FIXNUM) {
        if      (r1->tt==GURU_TT_FIXNUM) r0->i += r1->i;
#if GURU_USE_FLOAT
        else if (r1->tt==GURU_TT_FLOAT) {	// in case of Fixnum, Float
            r0->tt = GURU_TT_FLOAT;
            r0->f = r0->i + r1->f;
        }
        else console_na("Fixnum + ?");
    }
    else if (r0->tt==GURU_TT_FLOAT) {
        if      (r1->tt==GURU_TT_FIXNUM) r0->f += r1->i;
        else if (r1->tt==GURU_TT_FLOAT)	 r0->f += r1->f;
        else console_na("Float + ?");
#endif
    }
    else {    	// other case
    	op_send(vm, code, regs);			// should have already released regs[ra + n], ...
    }
    mrbc_release(r1);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_ADDI

  R(A) := R(A)+C (Syms[B]=:+)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_addi(guru_vm *vm, U32 code, mrbc_value *regs)
{
	int ra = GETARG_A(code);
	int rc = GETARG_C(code);

    mrbc_value *r0 = &regs[ra];

    if (r0->tt==GURU_TT_FIXNUM)     r0->i += rc;
#if GURU_USE_FLOAT
    else if (r0->tt==GURU_TT_FLOAT)	r0->f += rc;
#else
    else console_na("Float class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Execute OP_SUB

  R(A) := R(A)-R(A+1) (Syms[B]=:-,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_sub(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    mrbc_value *r0 = &regs[ra];
    mrbc_value *r1 = &regs[ra+1];

    if (r0->tt==GURU_TT_FIXNUM) {
        if      (r1->tt==GURU_TT_FIXNUM) 	r0->i -= r1->i;
#if GURU_USE_FLOAT
        else if (r1->tt==GURU_TT_FLOAT) {		// in case of Fixnum, Float
            r0->tt = GURU_TT_FLOAT;
            r0->f  = r0->i - r1->f;
        }
        else console_na("Fixnum - ?");
    }
    else if (r0->tt==GURU_TT_FLOAT) {
        if      (r1->tt==GURU_TT_FIXNUM)	r0->f -= r1->i;
        else if (r1->tt==GURU_TT_FLOAT)		r0->f -= r1->f;
        else console_na("Float - ?");
#endif
    }
    else {  // other case
    	op_send(vm, code, regs);
    }
    mrbc_release(r1);
	return 0;
}

//================================================================
/*!@brief
  Execute OP_SUBI

  R(A) := R(A)-C (Syms[B]=:-)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_subi(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rc = GETARG_C(code);

    mrbc_value *r0 = &regs[ra];

    if (r0->tt==GURU_TT_FIXNUM) 	r0->i -= rc;
#if GURU_USE_FLOAT
    else if (r0->tt==GURU_TT_FLOAT) r0->f -= rc;
#else
    else console_na("Float class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Execute OP_MUL

  R(A) := R(A)*R(A+1) (Syms[B]=:*)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_mul(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    mrbc_value *r0 = &regs[ra];
    mrbc_value *r1 = &regs[ra+1];

    if (r0->tt==GURU_TT_FIXNUM) {
        if      (r1->tt==GURU_TT_FIXNUM) 	r0->i *= r1->i;
#if GURU_USE_FLOAT
        else if (r1->tt==GURU_TT_FLOAT) {	// in case of Fixnum, Float
            r0->tt = GURU_TT_FLOAT;
            r0->f  = r0->i * r1->f;
        }
        else console_na("Fixnum * ?");
    }
    else if (r0->tt==GURU_TT_FLOAT) {
        if      (r1->tt==GURU_TT_FIXNUM) r0->f *= r1->i;
        else if (r1->tt==GURU_TT_FLOAT)  r0->f *= r1->f;
        else console_na("Float * ?");
#endif
    }
    else {   // other case
    	op_send(vm, code, regs);
    }
    mrbc_release(r1);
    return 0;
}

//================================================================
/*!@brief
  Execute OP_DIV

  R(A) := R(A)/R(A+1) (Syms[B]=:/)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_div(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    mrbc_value *r0 = &regs[ra];
    mrbc_value *r1 = &regs[ra+1];

    if (r0->tt==GURU_TT_FIXNUM) {
        if      (r1->tt==GURU_TT_FIXNUM) 	r0->i /= r1->i;
#if GURU_USE_FLOAT
        else if (r1->tt==GURU_TT_FLOAT) {		// in case of Fixnum, Float
            r0->tt = GURU_TT_FLOAT;
            r0->f  = r0->i / r1->f;
        }
        else console_na("Fixnum / ?");
    }
    else if (r0->tt==GURU_TT_FLOAT) {
        if      (r1->tt==GURU_TT_FIXNUM) 	r0->f /= r1->i;
        else if (r1->tt==GURU_TT_FLOAT)		r0->f /= r1->f;
        else console_na("Float / ?");
#endif
    }
    else {   // other case
    	op_send(vm, code, regs);
    }
    mrbc_release(r1);
    return 0;
}

//================================================================
/*!@brief
  Execute OP_EQ

  R(A) := R(A)==R(A+1)  (Syms[B]=:==,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_eq(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    mrbc_vtype tt = TT_BOOL(mrbc_compare(&regs[ra], &regs[ra+1])==0);

    _RA_T(tt, i=0);

    mrbc_release(&regs[ra+1]);

    return 0;
}

// macro for comparators
#define ncmp(r0, op, r1)								\
do {													\
	if ((r0)->tt==GURU_TT_FIXNUM) {						\
		if ((r1)->tt==GURU_TT_FIXNUM) {					\
			(r0)->tt = TT_BOOL((r0)->i op (r1)->i);		\
		}												\
		else if ((r1)->tt==GURU_TT_FLOAT) {				\
			(r0)->tt = TT_BOOL((r0)->i op (r1)->f);		\
		}												\
	}													\
	else if ((r0)->tt==GURU_TT_FLOAT) {					\
		if ((r1)->tt==GURU_TT_FIXNUM) {					\
			(r0)->tt = TT_BOOL((r0)->f op (r1)->i);		\
		}												\
		else if ((r1)->tt==GURU_TT_FLOAT) {				\
			(r0)->tt = TT_BOOL((r0)->f op (r1)->f);		\
		}												\
	}													\
	else {												\
		op_send(vm, code, regs);						\
	}													\
    mrbc_release(r1);									\
} while (0)

//================================================================
/*!@brief
  Execute OP_LT

  R(A) := R(A)<R(A+1)  (Syms[B]=:<,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_lt(guru_vm *vm, U32 code, mrbc_value *regs)
{
	int ra = GETARG_A(code);

	ncmp(&regs[ra], <, &regs[ra+1]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LE

  R(A) := R(A)<=R(A+1)  (Syms[B]=:<=,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ U32
op_le(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    ncmp(&regs[ra], <=, &regs[ra+1]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GT

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_gt(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    ncmp(&regs[ra], >, &regs[ra+1]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GE

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_ge(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    ncmp(&regs[ra], >=, &regs[ra+1]);

    return 0;
}

//================================================================
/*!@brief
  Create string object

  R(A) := str_dup(Lit(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_string(guru_vm *vm, U32 code, mrbc_value *regs)
{
#if GURU_USE_STRING
	int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    U32P p   = _vm_pool(vm, rb);
    U8P str  = (U8P)U8PADD(VM_IREP(vm), *p);
    mrbc_value ret = mrbc_string_new(str);

    if (ret.str==NULL) return vm->err = 255;			// ENOMEM

    _RA_V(ret);
#else
    console_na("String class");
#endif
    return 0;
}

//================================================================
/*!@brief
  String Catination

  str_cat(R(A),R(B))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_strcat(guru_vm *vm, U32 code, mrbc_value *regs)
{
#if GURU_USE_STRING
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);

    mrbc_value *pa  = &regs[ra];
    mrbc_value *pb  = &regs[rb];

    mrbc_sym sid  = name2symid((U8P)"to_s");			// from global symbol pool
    mrbc_proc *ma = mrbc_get_proc_by_symid(*pa, sid);
    mrbc_proc *mb = mrbc_get_proc_by_symid(*pb, sid);

    if (ma && IS_CFUNC(ma)) ma->func(pa, 0);
    if (mb && IS_CFUNC(mb)) mb->func(pb, 0);

    mrbc_value ret = mrbc_string_add(pa, pb);

    _RA_V(ret);

#else
    console_na("String class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Create Array object

  R(A) := ary_new(R(B),R(B+1)..R(B+C))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_array(guru_vm *vm, U32 code, mrbc_value *regs)
{
#if GURU_USE_ARRAY
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);
    int sz = sizeof(mrbc_value) * rc;

    mrbc_value ret = (mrbc_value)mrbc_array_new(rc);
    mrbc_array *h  = ret.array;
    mrbc_value *pb = &regs[rb];
    if (h==NULL) return vm->err = 255;	// ENOMEM

    MEMCPY((U8P)h->data, (U8P)pb, sz);
    MEMSET((U8P)pb, 0, sz);
    h->n = rc;

    _RA_V(ret);
#else
    console_na("Array class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Create Hash object

  R(A) := hash_new(R(B),R(B+1)..R(B+C))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_hash(guru_vm *vm, U32 code, mrbc_value *regs)
{
#if GURU_USE_ARRAY
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);
    int sz = sizeof(mrbc_value) * (rc<<1);				// size of k,v pairs

    mrbc_value ret = mrbc_hash_new(rc);
    mrbc_hash  *h  = ret.hash;
    mrbc_value *p  = &regs[rb];
    if (h==NULL) return vm->err = 255;					// ENOMEM

    MEMCPY((U8P)h->data, (U8P)p, sz);					// copy k,v pairs

    for (U32 i=0; i<(h->n=(rc<<1)); i++, p++) {
    	p->tt = GURU_TT_EMPTY;							// clean up call stack
    }
    _RA_V(ret);						                	// set return value on stack top
#else
    console_na("Hash class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Execute OP_RANGE

  R(A) := range_new(R(B),R(B+1),C)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_range(guru_vm *vm, U32 code, mrbc_value *regs)
{
#if GURU_USE_ARRAY
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);

    mrbc_value *pb = &regs[rb];
    mrbc_value ret = mrbc_range_new(pb, pb+1, rc);
    if (ret.range==NULL) return vm->err = 255;		// ENOMEM

    _RA_V(ret);						// release and  reassign
    mrbc_retain(pb);
    mrbc_retain(pb+1);

#else
    console_na("Range class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Execute OP_LAMBDA

  R(A) := lambda(SEQ[Bz],Cz)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_lambda(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_b(code);      		// sequence position in irep list
    // int c = GETARG_C(code);    		// TODO: Add flags support for OP_LAMBDA

    mrbc_proc *prc = (mrbc_proc *)mrbc_alloc_proc((U8P)"(lambda)");

    prc->irep = _vm_irep_list(vm, rb);
    prc->flag &= ~GURU_CFUNC;           // Ruby IREP

    _RA_T(GURU_TT_PROC, proc=prc);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_CLASS

  R(A) := newclass(R(A),Syms(B),R(A+1))
  Syms(B): class name
  R(A+1): super class

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_class(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);

    mrbc_class *super = (regs[ra+1].tt==GURU_TT_CLASS) ? regs[ra+1].cls : mrbc_class_object;
    const U8P  name   = _vm_symbol(vm, rb);
    mrbc_class *cls   = (mrbc_class *)mrbc_define_class(name, super);

    _RA_T(GURU_TT_CLASS, cls=cls);
    return 0;
}

//================================================================
/*!@brief
  Execute OP_EXEC

  R(A) := blockexec(R(A),SEQ[Bx])

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_exec(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    mrbc_value rcv = regs[ra];							// receiver

    _push_state(vm, 0);									// push call stack

    vm->state->irep  = _vm_irep_list(vm, rb);			// fetch designated irep
    vm->state->pc 	 = 0;								// switch context to callee
    vm->state->reg 	 += ra;								// shift regfile (for local stack)
    vm->state->klass = mrbc_get_class_by_object(&rcv);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_METHOD

  R(A).newmethod(Syms(B),R(A+1))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_method(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);

    if (regs[ra].tt != GURU_TT_CLASS) {
    	console_str("?op_method");
    	return 0;
    }

    mrbc_class 	*cls  = regs[ra].cls;
    mrbc_proc 	*prc  = regs[ra+1].proc;
    mrbc_sym	sid   = _vm_symid(vm, rb);

    MUTEX_LOCK(_mutex_op);

    // check same name method
    mrbc_proc 	*p  = cls->vtbl;
    void 		*pp = &cls->vtbl;
    while (p != NULL) {						// walk through vtable
    	if (p->sym_id == sid) break;
    	pp = &p->next;
    	p  = p->next;
    }
    if (p) {	// found?
    	*((mrbc_proc**)pp) = p->next;
    	if (!IS_CFUNC(p)) {				// a p->func a Ruby function (aka IREP)
    		mrbc_value v = { .tt = GURU_TT_PROC };
    		v.proc = p;
    		mrbc_release(&v);
        }
    }

    // add proc to class
    prc->sym_id = sid;
    prc->flag   &= ~GURU_CFUNC;

    prc->next   = cls->vtbl;				// add to top of vtable
    cls->vtbl   = prc;

    MUTEX_FREE(_mutex_op);

    regs[ra+1].tt = GURU_TT_EMPTY;

    return 0;
}

//================================================================
/*!@brief
  Execute OP_TCLASS

  R(A) := target_class

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval 0  No error.
*/
__GURU__ int
op_tclass(guru_vm *vm, U32 code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    _RA_T(GURU_TT_CLASS, cls=vm->state->klass);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_STOP and OP_ABORT

  stop VM (OP_STOP)
  stop VM without release memory (OP_ABORT)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regfile + vm->state->reg
  @retval -1  No error and exit from vm.
*/
__GURU__ int
op_stop(guru_vm *vm, U32 code, mrbc_value *regs)
{
	vm->run = 0;
    return -1;		// exit guru_op loop
}

__GURU__ int
op_abort(guru_vm *vm, U32 code, mrbc_value *regs)
{
	return -1;		// exit guru_op loop
}

__GURU__ int
guru_op(guru_vm *vm)
{
	if (threadIdx.x != 0) return 0;	// TODO: multi-thread

	U32 code   = GET_BYTECODE(vm);
	U32 opcode = GET_OPCODE(code);
	mrbc_value *regs  = vm->state->reg;
	int ret;

    vm->state->pc++;				// advance program counter, ready for next cycle
    switch (opcode) {
    // LOAD,STORE
    case OP_LOADL:      ret = op_loadl     (vm, code, regs); break;
    case OP_LOADI:      ret = op_loadi     (vm, code, regs); break;
    case OP_LOADSYM:    ret = op_loadsym   (vm, code, regs); break;
    case OP_LOADNIL:    ret = op_loadnil   (vm, code, regs); break;
    case OP_LOADSELF:   ret = op_loadself  (vm, code, regs); break;
    case OP_LOADT:      ret = op_loadt     (vm, code, regs); break;
    case OP_LOADF:      ret = op_loadf     (vm, code, regs); break;
    // VARIABLES
    case OP_GETGLOBAL:  ret = op_getglobal (vm, code, regs); break;
    case OP_SETGLOBAL:  ret = op_setglobal (vm, code, regs); break;
    case OP_GETIV:      ret = op_getiv     (vm, code, regs); break;
    case OP_SETIV:      ret = op_setiv     (vm, code, regs); break;
    case OP_GETCONST:   ret = op_getconst  (vm, code, regs); break;
    case OP_SETCONST:   ret = op_setconst  (vm, code, regs); break;
    case OP_GETUPVAR:   ret = op_getupvar  (vm, code, regs); break;
    case OP_SETUPVAR:   ret = op_setupvar  (vm, code, regs); break;
    // BRANCH
    case OP_JMP:        ret = op_jmp       (vm, code, regs); break;
    case OP_JMPIF:      ret = op_jmpif     (vm, code, regs); break;
    case OP_JMPNOT:     ret = op_jmpnot    (vm, code, regs); break;
    case OP_SEND:       ret = op_send      (vm, code, regs); break;
    case OP_SENDB:      ret = op_send      (vm, code, regs); break;  // reuse
    case OP_CALL:       ret = op_call      (vm, code, regs); break;
    case OP_ENTER:      ret = op_enter     (vm, code, regs); break;
    case OP_RETURN:     ret = op_return    (vm, code, regs); break;
    case OP_BLKPUSH:    ret = op_blkpush   (vm, code, regs); break;
    // ALU
    case OP_MOVE:       ret = op_move      (vm, code, regs); break;
    case OP_ADD:        ret = op_add       (vm, code, regs); break;
    case OP_ADDI:       ret = op_addi      (vm, code, regs); break;
    case OP_SUB:        ret = op_sub       (vm, code, regs); break;
    case OP_SUBI:       ret = op_subi      (vm, code, regs); break;
    case OP_MUL:        ret = op_mul       (vm, code, regs); break;
    case OP_DIV:        ret = op_div       (vm, code, regs); break;
    case OP_EQ:         ret = op_eq        (vm, code, regs); break;
    case OP_LT:         ret = op_lt        (vm, code, regs); break;
    case OP_LE:         ret = op_le        (vm, code, regs); break;
    case OP_GT:         ret = op_gt        (vm, code, regs); break;
    case OP_GE:         ret = op_ge        (vm, code, regs); break;
#if GURU_USE_STRING
    case OP_STRING:     ret = op_string    (vm, code, regs); break;
    case OP_STRCAT:     ret = op_strcat    (vm, code, regs); break;
#endif
#if GURU_USE_ARRAY
    case OP_ARRAY:      ret = op_array     (vm, code, regs); break;
    case OP_HASH:       ret = op_hash      (vm, code, regs); break;
    case OP_RANGE:      ret = op_range     (vm, code, regs); break;
#endif
    // BRANCH
    case OP_LAMBDA:     ret = op_lambda    (vm, code, regs); break;
    case OP_CLASS:      ret = op_class     (vm, code, regs); break;
    case OP_EXEC:       ret = op_exec      (vm, code, regs); break;
    case OP_METHOD:     ret = op_method    (vm, code, regs); break;
    case OP_TCLASS:     ret = op_tclass    (vm, code, regs); break;
    // EXEC
    case OP_STOP:       ret = op_stop      (vm, code, regs); break;
    case OP_ABORT:      ret = op_abort     (vm, code, regs); break;  	// reuse
    case OP_NOP:        ret = op_nop       (vm, code, regs); break;
    default:
    	console_str("?OP=");
    	console_int(opcode);
    	console_str("\n");
    	ret = 0;
    	break;
    }
    return ret;
}

