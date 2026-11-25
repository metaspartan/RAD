use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    pub rope_type: String,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position_embeddings: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub swiglu_limit: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rope_scaling: RopeScalingConfig,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
    pub sliding_window: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f32,
    pub kv_source_layer_ids: Vec<usize>,
    pub index_source_layer_ids: Vec<usize>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source_layer_id: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub engram_layer_ids: Vec<usize>,
    pub engram_max_ngram_size: usize,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    #[serde(default)]
    pub dspark_block_size: usize,
    #[serde(default)]
    pub dspark_markov_rank: usize,
    #[serde(default)]
    pub dspark_n_routed_experts: usize,
    #[serde(default)]
    pub dspark_noise_token_id: i32,
    #[serde(default)]
    pub dspark_num_experts_per_tok: usize,
    #[serde(default)]
    pub dspark_target_layer_ids: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct DeepseekV41Config {
    pub bos_token_id: i32,
    pub eos_token_id: i32,
    pub pad_token_id: i32,
    pub text_config: TextConfig,
}

impl DeepseekV41Config {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(model_dir.join("config.json"))?;
        Ok(serde_json::from_str(&raw)?)
    }
}
