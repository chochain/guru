/*! @file
  @brief
  GURU microcode unit (instruction + dispatcher)

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_OPCODE_H_
#define GURU_SRC_OPCODE_H_

#include "guru.h"
#include "vm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAXARG_Bx                   (0xffff)
#define MAXARG_sBx                  (MAXARG_Bx>>1)
    
#define GET_OPCODE(i)              ((i) & 0x7f)

// common OPs
#define GETARG_A(i)                (((i) >> 23) & 0x1ff)
#define GETARG_B(i)                (((i) >> 14) & 0x1ff)
#define GETARG_C(i)                (((i) >>  7) & 0x7f)
#define GETARG_Bx(i)               (((i) >>  7) & 0xffff)

#define GETARG_sBx(i)              (GETARG_Bx(i)-MAXARG_sBx)

// special format for JUMP (25-bit address)
#define GETARG_Ax(i)               (((i) >>  7) & 0x1ffffff)

#define GETARG_UNPACK_b(i,n1,n2)   ((((i)) >> (7+(n2))) & (((1<<(n1))-1)))
#define GETARG_UNPACK_c(i,n1,n2)   ((int)((((mrb_code)(i)) >> 7) & (((1<<(n2))-1))))
#define GETARG_b(i)                GETARG_UNPACK_b(i,14,2)

#define MKOPCODE(op)               ((op & 0x7f)<<24)
#define MKARG_A(c)                 ((c & 0xff)<<1 | (c & 0x01)>>8)
#define MKARG_B(c)                 ((c & 0x1fc)<<6 | (c & 0x03)<<22)
#define MKARG_C(c)                 ((c & 0x7e)<<15 | (c & 0x01)<<31)

// the following are not implemented
#define MKARG_Bx(v)                ((mrb_code)((v) & 0xffff) << 7)
#define MKARG_sBx(v)               MKARG_Bx((v)+MAXARG_sBx)
#define MKARG_Ax(v)                ((mrb_code)((v) & 0x1ffffff) << 7)
#define MKARG_PACK(b,n1,c,n2)      ((((b) & ((1<<n1)-1)) << (7+n2))|(((c) & ((1<<n2)-1)) << 7))
#define MKARG_bc(b,c)              MKARG_PACK(b,14,c,2)

#define MKOP_A(op,a)               (MKOPCODE(op)|MKARG_A(a))
#define MKOP_AB(op,a,b)            (MKOP_A(op,a)|MKARG_B(b))
#define MKOP_ABC(op,a,b,c)         (MKOP_AB(op,a,b)|MKARG_C(c))
#define MKOP_ABx(op,a,bx)          (MKOP_A(op,a)|MKARG_Bx(bx))
#define MKOP_Bx(op,bx)             (MKOPCODE(op)|MKARG_Bx(bx))
#define MKOP_sBx(op,sbx)           (MKOPCODE(op)|MKARG_sBx(sbx))
#define MKOP_AsBx(op,a,sbx)        (MKOP_A(op,a)|MKARG_sBx(sbx))
#define MKOP_Ax(op,ax)             (MKOPCODE(op)|MKARG_Ax(ax))
#define MKOP_Abc(op,a,b,c)         (MKOP_A(op,a)|MKARG_bc(b,c))


//================================================================
/*!@brief

 */
enum OPCODE {
    OP_NOP      = 0,
    OP_MOVE,     /*  A B     R(A) := R(B)                                    */
    OP_LOADL,    /*  A Bx    R(A) := Pool(Bx)                                */
    OP_LOADI,    /*  A sBx   R(A) := sBx                                     */
    OP_LOADSYM,  /*  A Bx    R(A) := Syms(Bx)                                */
    OP_LOADNIL,  /*  A       R(A) := nil                                     */
    OP_LOADSELF, /*  A       R(A) := self                                    */
    OP_LOADT,    /*  A       R(A) := true                                    */
    OP_LOADF,    /*  A       R(A) := false                                   */

    // 0x09
    OP_GETGLOBAL ,/* A Bx    R(A) := getglobal(Syms(Bx))                     */
    OP_SETGLOBAL, /* A Bx    setglobal(Syms(Bx), R(A))                       */
    OP_GETSPECIAL,/**A Bx    R(A) := Special[Bx]                             */
    OP_SETSPECIAL,/**A Bx    Special[Bx] := R(A)                             */
    OP_GETIV,     /* A Bx    R(A) := ivget(Syms(Bx))                         */
    OP_SETIV,     /* A Bx    ivset(Syms(Bx),R(A))                            */
    OP_GETCV,     /**A Bx    R(A) := cvget(Syms(Bx))                         */
    OP_SETCV,     /**A Bx    cvset(Syms(Bx),R(A))                            */
    OP_GETCONST,  /* A Bx    R(A) := constget(Syms(Bx))                      */
    OP_SETCONST,  /* A Bx    constset(Syms(Bx),R(A))                         */
    OP_GETMCNST,  /**A Bx    R(A) := R(A)::Syms(Bx)                          */
    OP_SETMCNST,  /**A Bx    R(A+1)::Syms(Bx) := R(A)                        */
    OP_GETUPVAR,  /* A B C   R(A) := uvget(B,C)                              */
    OP_SETUPVAR,  /* A B C   uvset(B,C,R(A))                                 */

    // 0x17
    OP_JMP,       /* sBx     pc+=sBx                                         */
    OP_JMPIF,     /* A sBx   if R(A) pc+=sBx                                 */
    OP_JMPNOT,    /* A sBx   if !R(A) pc+=sBx                                */
    OP_ONERR,     /**sBx     rescue_push(pc+sBx)                             */
    OP_RESCUE,    /**A B C   if A (if C exc=R(A) else R(A) := exc);
                             if B R(B) := exc.isa?(R(B)); clear(exc)         */
    OP_POPERR,    /**A       A.times{rescue_pop()}                           */
    OP_RAISE,     /**A       raise(R(A))                                     */
    OP_EPUSH,     /**Bx      ensure_push(SEQ[Bx])                            */
    OP_EPOP,      /**A       A.times{ensure_pop().call}                      */

    // 0x20
    OP_SEND,      /* A B C   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C))    */
    OP_SENDB,     /* A B C   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C),&R(A+C+1))*/
    OP_FSEND,     /**A B C   R(A) := fcall(R(A),Syms(B),R(A+1),...,R(A+C-1)) */
    OP_CALL,      /* A       R(A) := self.call(frame.argc, frame.argv)       */
    OP_SUPER,     /**A C     R(A) := super(R(A+1),... ,R(A+C+1))             */
    OP_ARGARY,    /**A Bx    R(A) := argument array (16=6:1:5:4)             */
    OP_ENTER,     /* Ax      arg setup according to flags (23=5:5:1:5:5:1:1) */
    OP_KARG,      /**A B C   R(A) := kdict[Syms(B)]; if C kdict.rm(Syms(B))  */
    OP_KDICT,     /**A C     R(A) := kdict                                   */
    OP_RETURN,    /* A B     return R(A) (B=normal,in-block return/break)    */
    OP_TAILCALL,  /**A B C   return call(R(A),Syms(B),*R(C))                 */
    OP_BLKPUSH,   /* A Bx    R(A) := block (16=6:1:5:4)                      */

    // 0x2c
    OP_ADD,       /* A B C   R(A) := R(A)+R(A+1) (Syms[B]=:+,C=1)            */
    OP_ADDI,      /* A B C   R(A) := R(A)+C (Syms[B]=:+)                     */
    OP_SUB,       /* A B C   R(A) := R(A)-R(A+1) (Syms[B]=:-,C=1)            */
    OP_SUBI,      /* A B C   R(A) := R(A)-C (Syms[B]=:-)                     */
    OP_MUL,       /* A B C   R(A) := R(A)*R(A+1) (Syms[B]=:*,C=1)            */
    OP_DIV,       /* A B C   R(A) := R(A)/R(A+1) (Syms[B]=:/,C=1)            */
    OP_EQ,        /* A B C   R(A) := R(A)==R(A+1) (Syms[B]=:==,C=1)          */
    OP_LT,        /* A B C   R(A) := R(A)<R(A+1)  (Syms[B]=:<,C=1)           */
    OP_LE,        /* A B C   R(A) := R(A)<=R(A+1) (Syms[B]=:<=,C=1)          */
    OP_GT,        /* A B C   R(A) := R(A)>R(A+1)  (Syms[B]=:>,C=1)           */
    OP_GE,        /* A B C   R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)          */

    // 0x37
    OP_ARRAY,     /* A B C   R(A) := ary_new(R(B),R(B+1)..R(B+C))            */
    OP_ARYCAT,    /**A B     ary_cat(R(A),R(B))                              */
    OP_ARYPUSH,   /**A B     ary_push(R(A),R(B))                             */
    OP_AREF,      /**A B C   R(A) := R(B)[C]                                 */
    OP_ASET,      /**A B C   R(B)[C] := R(A)                                 */
    OP_APOST,     /** A B C   *R(A),R(A+1)..R(A+C) := R(A)[B..]              */

    OP_STRING,    /* A Bx    R(A) := str_dup(Lit(Bx))                        */
    OP_STRCAT,    /* A B     str_cat(R(A),R(B))                              */

    OP_HASH,      /* A B C   R(A) := hash_new(R(B),R(B+1)..R(B+C))           */
    OP_LAMBDA,    /* A Bz Cz R(A) := lambda(SEQ[Bz],Cz)                      */
    OP_RANGE,     /* A B C   R(A) := range_new(R(B),R(B+1),C)                */

    // 0x42
    OP_OCLASS,    /**A       R(A) := ::Object                                */
    OP_CLASS,     /* A B     R(A) := newclass(R(A),Syms(B),R(A+1))           */
    OP_MODULE,    /**A B     R(A) := newmodule(R(A),Syms(B))                 */
    OP_EXEC,      /* A Bx    R(A) := blockexec(R(A),SEQ[Bx])                 */
    OP_METHOD,    /* A B     R(A).newmethod(Syms(B),R(A+1))                  */
    OP_SCLASS,    /**A B     R(A) := R(B).singleton_class                    */
    OP_TCLASS,    /* A       R(A) := target_class                            */
    OP_DEBUG,     /**A B C   print R(A),R(B),R(C)                            */

    // 0x4a
    OP_STOP,      /*         stop VM                                         */
    OP_ERR,       /**Bx      raise RuntimeError with message Lit(Bx)         */
    OP_RSVD1,     /*         reserved instruction #1                         */
    OP_RSVD2,     /*         reserved instruction #2                         */
    OP_RSVD3,     /*         reserved instruction #3                         */
    OP_RSVD4,     /*         reserved instruction #4                         */
    OP_ABORT     = 0x50,  // using OP_ABORT inside mruby/c only
};

__GURU__ void guru_op(guru_vm *vm);

#ifdef __cplusplus
}
#endif
#endif
