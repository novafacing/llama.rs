use lazy_static::lazy_static;
use libc;
use std::collections::HashMap;

const LLAMA_FILE_MAGIC_GGJT: u32 = 0x67676a74; // 'ggjt'
const LLAMA_FILE_MAGIC_GGLA: u32 = 0x67676c61; // 'ggla'
const LLAMA_FILE_MAGIC_GGMF: u32 = 0x67676d66; // 'ggmf'
const LLAMA_FILE_MAGIC_GGML: u32 = 0x67676d6c; // 'ggml'
const LLAMA_FILE_MAGIC_GGSN: u32 = 0x6767736e; // 'ggsn'
const LLAMA_FILE_VERSION: u32 = 3;
const LLAMA_FILE_MAGIC: u32 = LLAMA_FILE_MAGIC_GGJT;
const LLAMA_FILE_MAGIC_UNVERSIONED: u32 = LLAMA_FILE_MAGIC_GGML;
const LLAMA_SESSION_MAGIC: u32 = LLAMA_FILE_MAGIC_GGSN;
const LLAMA_SESSION_VERSION: u32 = 1;

type llama_token = i32;

#[repr(C)]
pub struct llama_token_data {
    pub id: llama_token,
    pub logit: f32,
    pub p: f32,
}

#[repr(C)]
pub struct llama_token_data_array {
    pub data: *mut llama_token_data,
    pub size: u64,
    pub sorted: bool,
}

type llama_progress_callback = extern "C" fn(progress: f32, ctx: *mut libc::c_void);

#[repr(C)]
pub struct llama_context_params {
    pub n_ctx: i32,
    pub n_gpu_layers: i32,
    pub seed: i32,
    pub f16_kv: bool,
    pub logits_all: bool,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub embedding: bool,
    pub progress_callback: llama_progress_callback,
    pub progress_callback_user_data: *mut libc::c_void,
}

const LLAMA_FTYPE_ALL_F32: u32 = 0;
const LLAMA_FTYPE_MOSTLY_F16: u32 = 1;
const LLAMA_FTYPE_MOSTLY_Q4_0: u32 = 2;
const LLAMA_FTYPE_MOSTLY_Q4_1: u32 = 3;
const LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16: u32 = 4;
const LLAMA_FTYPE_MOSTLY_Q8_0: u32 = 7;
const LLAMA_FTYPE_MOSTLY_Q5_0: u32 = 8;
const LLAMA_FTYPE_MOSTLY_Q5_1: u32 = 9;

const MODEL_UNKNOWN: i32 = 0;
const MODEL_7B: i32 = 1;
const MODEL_13B: i32 = 2;
const MODEL_30B: i32 = 3;
const MODEL_65B: i32 = 4;

lazy_static! {
    static ref MEM_REQ_SCRATCH0: HashMap<i32, usize> = {
        let mut m = HashMap::new();
        m.insert(MODEL_7B, 512 * 1024 * 1024);
        m.insert(MODEL_13B, 512 * 1024 * 1024);
        m.insert(MODEL_30B, 512 * 1024 * 1024);
        m.insert(MODEL_65B, 1024 * 1024 * 1024);
        m
    };
    static ref MEM_REQ_SCRATCH1: HashMap<i32, usize> = {
        let mut m = HashMap::new();
        m.insert(MODEL_7B, 512 * 1024 * 1024);
        m.insert(MODEL_13B, 512 * 1024 * 1024);
        m.insert(MODEL_30B, 512 * 1024 * 1024);
        m.insert(MODEL_65B, 1024 * 1024 * 1024);
        m
    };
    static ref MEM_REQ_KV_SELF: HashMap<i32, usize> = {
        let mut m = HashMap::new();
        m.insert(MODEL_7B, 1026 * 1024 * 1024);
        m.insert(MODEL_13B, 1608 * 1024 * 1024);
        m.insert(MODEL_30B, 3124 * 1024 * 1024);
        m.insert(MODEL_65B, 5120 * 1024 * 1024);
        m
    };
    static ref MEM_REQ_EVAL: HashMap<i32, usize> = {
        let mut m = HashMap::new();
        m.insert(MODEL_7B, 768 * 1024 * 1024);
        m.insert(MODEL_13B, 1024 * 1024 * 1024);
        m.insert(MODEL_30B, 1280 * 1024 * 1024);
        m.insert(MODEL_65B, 1536 * 1024 * 1024);
        m
    };
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LlamaHyperParams {
    pub n_vocab: u32,
    pub n_ctx: u32,
    pub n_embd: u32,
    pub n_mult: u32,
    pub n_head: u32,
    pub n_layer: u32,
    pub n_rot: u32,
    pub ftype: u32,
}

impl Default for LlamaHyperParams {
    fn default() -> Self {
        LlamaHyperParams {
            n_vocab: 32000,
            n_ctx: 512,
            n_embd: 256,
            n_mult: 256,
            n_head: 32,
            n_layer: 32,
            n_rot: 64,
            ftype: LLAMA_FTYPE_MOSTLY_F16,
        }
    }
}
