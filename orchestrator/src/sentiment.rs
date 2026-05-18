use anyhow::Result;
use ort::{inputs, session::Session, value::Tensor};
use std::sync::Mutex;
use tokenizers::Tokenizer;

const MAX_LEN: usize = 128;

pub struct SentimentClassifier {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pub threshold: f32,
}

impl SentimentClassifier {
    pub fn load(model_dir: &str, threshold: f32) -> Result<Self> {
        let session =
            Mutex::new(Session::builder()?.commit_from_file(format!("{model_dir}/model.onnx"))?);

        let tokenizer = Tokenizer::from_file(format!("{model_dir}/tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;

        tracing::info!(target: "sentiment", model_dir, threshold, "sentiment classifier loaded");
        Ok(Self {
            session,
            tokenizer,
            threshold,
        })
    }

    /// Returns true when the text carries a negative sentiment score at or above the threshold.
    pub fn is_negative(&self, text: &str) -> bool {
        let started = std::time::Instant::now();
        match self.negative_score(text) {
            Ok(score) => {
                tracing::debug!(target: "sentiment", score, threshold = self.threshold);
                let negative = score_exceeds_threshold(score, self.threshold);
                crate::telemetry::record_sentiment(negative, started.elapsed());
                negative
            }
            Err(e) => {
                tracing::warn!(target: "sentiment", "inference error: {e}");
                false
            }
        }
    }

    fn negative_score(&self, text: &str) -> Result<f32> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;

        let len = encoding.get_ids().len().min(MAX_LEN);
        let ids: Vec<i64> = encoding.get_ids()[..len]
            .iter()
            .map(|&x| x as i64)
            .collect();
        let mask: Vec<i64> = encoding.get_attention_mask()[..len]
            .iter()
            .map(|&x| x as i64)
            .collect();

        let input_ids_t = Tensor::<i64>::from_array(([1usize, len], ids))?;
        let attn_mask_t = Tensor::<i64>::from_array(([1usize, len], mask))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;
        let outputs = session.run(inputs![
            "input_ids" => input_ids_t,
            "attention_mask" => attn_mask_t,
        ])?;

        // DistilBERT-SST2: logits[0] = NEGATIVE, logits[1] = POSITIVE
        let (_shape, data) = outputs["logits"].try_extract_tensor::<f32>()?;
        Ok(softmax_negative_prob(data[0], data[1]))
    }
}

/// Softmax over two logits returning P(NEGATIVE).
/// Subtracts max for numerical stability before exponentiating.
pub(crate) fn softmax_negative_prob(neg_logit: f32, pos_logit: f32) -> f32 {
    let max = neg_logit.max(pos_logit);
    let e_neg = (neg_logit - max).exp();
    let e_pos = (pos_logit - max).exp();
    e_neg / (e_neg + e_pos)
}

pub(crate) fn score_exceeds_threshold(score: f32, threshold: f32) -> bool {
    score >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ── softmax_negative_prob ───────────────────────────────────────────

    #[test]
    fn dominant_negative_logit_yields_high_probability() {
        let p = softmax_negative_prob(10.0, -10.0);
        assert!(p > 0.999, "expected p ≈ 1.0, got {p}");
    }

    #[test]
    fn dominant_positive_logit_yields_low_probability() {
        let p = softmax_negative_prob(-10.0, 10.0);
        assert!(p < 0.001, "expected p ≈ 0.0, got {p}");
    }

    #[test]
    fn equal_logits_yield_half_probability() {
        let p = softmax_negative_prob(0.0, 0.0);
        assert!(approx_eq(p, 0.5), "expected p ≈ 0.5, got {p}");
    }

    #[test]
    fn probability_is_bounded_zero_to_one() {
        for (neg, pos) in [(-100.0f32, 100.0), (100.0, -100.0), (0.0, 0.0), (1.5, 0.3)] {
            let p = softmax_negative_prob(neg, pos);
            assert!(
                p >= 0.0 && p <= 1.0,
                "p={p} out of [0,1] for neg={neg}, pos={pos}"
            );
        }
    }

    #[test]
    fn probability_is_monotone_in_negative_logit() {
        // increasing neg_logit while holding pos_logit constant must increase P(negative)
        let p_low = softmax_negative_prob(0.0, 0.0);
        let p_mid = softmax_negative_prob(1.0, 0.0);
        let p_high = softmax_negative_prob(5.0, 0.0);
        assert!(p_low < p_mid && p_mid < p_high);
    }

    // ── score_exceeds_threshold ─────────────────────────────────────────

    #[test]
    fn score_at_threshold_is_classified_negative() {
        assert!(score_exceeds_threshold(0.70, 0.70));
    }

    #[test]
    fn score_just_below_threshold_is_not_negative() {
        assert!(!score_exceeds_threshold(0.699, 0.70));
    }

    #[test]
    fn score_above_threshold_is_negative() {
        assert!(score_exceeds_threshold(0.95, 0.70));
    }

    #[test]
    fn zero_threshold_always_classifies_negative() {
        assert!(score_exceeds_threshold(0.0, 0.0));
        assert!(score_exceeds_threshold(0.5, 0.0));
        assert!(score_exceeds_threshold(1.0, 0.0));
    }

    #[test]
    fn threshold_above_one_never_classifies_negative() {
        assert!(!score_exceeds_threshold(1.0, 1.1));
    }

    // ── combined: softmax output drives threshold correctly ─────────────

    #[test]
    fn strongly_negative_logits_trigger_at_standard_threshold() {
        let score = softmax_negative_prob(5.0, -5.0);
        assert!(score_exceeds_threshold(score, 0.70));
    }

    #[test]
    fn strongly_positive_logits_do_not_trigger_at_standard_threshold() {
        let score = softmax_negative_prob(-5.0, 5.0);
        assert!(!score_exceeds_threshold(score, 0.70));
    }
}
