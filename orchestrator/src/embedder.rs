use anyhow::Result;
use ort::{inputs, session::Session, value::Tensor};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

const MAX_LEN: usize = 512;

pub struct Embedder {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
}

impl Embedder {
    pub fn load(model_dir: &str) -> Result<Self> {
        let session = Arc::new(Mutex::new(
            Session::builder()?.commit_from_file(format!("{model_dir}/model.onnx"))?,
        ));
        let tokenizer = Arc::new(
            Tokenizer::from_file(format!("{model_dir}/tokenizer.json"))
                .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?,
        );
        tracing::info!(target: "embedder", model_dir, "embedding model loaded");
        Ok(Self { session, tokenizer })
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let session = Arc::clone(&self.session);
        let tokenizer = Arc::clone(&self.tokenizer);
        let text = text.to_string();
        tokio::task::spawn_blocking(move || embed_sync(session, &tokenizer, &text))
            .await
            .map_err(|e| anyhow::anyhow!("embed task panicked: {e}"))?
    }
}

fn embed_sync(session: Arc<Mutex<Session>>, tokenizer: &Tokenizer, text: &str) -> Result<Vec<f32>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;

    let len = encoding.get_ids().len().min(MAX_LEN);
    let ids: Vec<i64> = encoding.get_ids()[..len].iter().map(|&x| x as i64).collect();
    let mask: Vec<i64> = encoding.get_attention_mask()[..len]
        .iter()
        .map(|&x| x as i64)
        .collect();

    let input_ids_t = Tensor::<i64>::from_array(([1usize, len], ids))?;
    let attn_mask_t = Tensor::<i64>::from_array(([1usize, len], mask.clone()))?;

    let mut session = session.lock().map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;
    let outputs = session.run(inputs![
        "input_ids" => input_ids_t,
        "attention_mask" => attn_mask_t,
    ])?;

    // last_hidden_state: [batch=1, seq_len, hidden_size]
    let (shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
    let seq_len = shape[1] as usize;
    let hidden_size = shape[2] as usize;

    Ok(mean_pool_and_normalize(seq_len, hidden_size, data, &mask))
}

/// Mean-pools token embeddings over non-padding positions then L2-normalises.
/// `hidden_flat` is the raw `[1, seq_len, hidden_size]` tensor flattened to a slice.
pub(crate) fn mean_pool_and_normalize(
    seq_len: usize,
    hidden_size: usize,
    hidden_flat: &[f32],
    mask: &[i64],
) -> Vec<f32> {
    let mut pooled = vec![0.0f32; hidden_size];
    let mask_sum: f32 = mask.iter().map(|&x| x as f32).sum();

    for i in 0..seq_len {
        if mask[i] == 1 {
            for j in 0..hidden_size {
                pooled[j] += hidden_flat[i * hidden_size + j];
            }
        }
    }

    let denom = mask_sum.max(1e-9);
    for v in &mut pooled {
        *v /= denom;
    }

    let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for v in &mut pooled {
        *v /= norm;
    }

    pooled
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    // hidden_flat layout: row-major [seq_len, hidden_size]
    // e.g. 2 tokens × 2 dims: [tok0_dim0, tok0_dim1, tok1_dim0, tok1_dim1]

    #[test]
    fn uniform_mask_averages_all_tokens() {
        // token 0 = [1, 0], token 1 = [0, 1]  →  mean = [0.5, 0.5]
        let flat = vec![1.0f32, 0.0, 0.0, 1.0];
        let mask = vec![1i64, 1];
        let result = mean_pool_and_normalize(2, 2, &flat, &mask);
        assert_eq!(result.len(), 2);
        assert!(approx_eq(result[0], result[1]), "dims should be equal after symmetric pool");
        assert!(approx_eq(l2_norm(&result), 1.0));
    }

    #[test]
    fn masked_padding_tokens_are_excluded() {
        // token 0 = [1, 0], token 1 = [0, 1], token 2 (pad) = [99, 99]
        let flat = vec![1.0f32, 0.0, 0.0, 1.0, 99.0, 99.0];
        let mask = vec![1i64, 1, 0];
        let result = mean_pool_and_normalize(3, 2, &flat, &mask);
        // padding must not influence the output
        assert!(approx_eq(result[0], result[1]));
        assert!(approx_eq(l2_norm(&result), 1.0));
    }

    #[test]
    fn output_is_unit_vector_for_arbitrary_input() {
        let flat = vec![3.0f32, 0.0, 0.0, 4.0, 1.0, 2.0];
        let mask = vec![1i64, 1, 1];
        let result = mean_pool_and_normalize(3, 2, &flat, &mask);
        assert!(approx_eq(l2_norm(&result), 1.0));
    }

    #[test]
    fn single_unmasked_token_normalised_correctly() {
        // [3, 4] → norm 5 → normalised [0.6, 0.8]
        let flat = vec![3.0f32, 4.0];
        let mask = vec![1i64];
        let result = mean_pool_and_normalize(1, 2, &flat, &mask);
        assert!(approx_eq(result[0], 0.6));
        assert!(approx_eq(result[1], 0.8));
    }

    #[test]
    fn hidden_size_preserved_in_output() {
        let hidden_size = 8;
        let flat = vec![1.0f32; 4 * hidden_size];
        let mask = vec![1i64; 4];
        let result = mean_pool_and_normalize(4, hidden_size, &flat, &mask);
        assert_eq!(result.len(), hidden_size);
    }

    #[test]
    fn all_padding_mask_does_not_panic() {
        let flat = vec![1.0f32, 2.0, 3.0, 4.0];
        let mask = vec![0i64, 0];
        let result = mean_pool_and_normalize(2, 2, &flat, &mask);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn only_masked_token_contributes() {
        // token 0 (pad) = [0, 0], token 1 (real) = [1, 0]
        let flat = vec![0.0f32, 0.0, 1.0, 0.0];
        let mask = vec![0i64, 1];
        let result = mean_pool_and_normalize(2, 2, &flat, &mask);
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[1], 0.0));
    }
}
