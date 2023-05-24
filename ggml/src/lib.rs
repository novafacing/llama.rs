#![allow(dead_code)]
#![allow(mutable_transmutes)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_assignments)]
#![allow(unused_mut)]
#![feature(core_intrinsics)]
#![feature(extern_types)]
#![feature(register_tool)]
#![feature(stdsimd)]
#![register_tool(c2rust)]
#![feature(repr_simd)]

extern crate libc;
pub mod ggml;
