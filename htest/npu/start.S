.section .text.start
.global _start
_start:
    li sp, 0x80100000
    call main
1:  j 1b
