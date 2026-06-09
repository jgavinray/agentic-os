use metrics::{describe_counter, describe_gauge, describe_histogram};

pub(crate) fn describe_model_context_metrics() {
    describe_counter!(
        "embedder_inferences_total",
        "Embedding model inference calls."
    );
    describe_histogram!(
        "embedder_inference_duration_seconds",
        "Embedding model inference latency in seconds."
    );
    describe_histogram!("embedder_input_tokens", "Tokenized embedder input length.");
    describe_counter!(
        "sentiment_inferences_total",
        "Sentiment classifier inferences by verdict."
    );
    describe_histogram!(
        "sentiment_inference_duration_seconds",
        "Sentiment classifier inference latency in seconds."
    );
    describe_counter!("summarizer_ticks_total", "Summarizer loop ticks.");
    describe_counter!(
        "summarizer_candidates_found_total",
        "Summarizer candidates found by target level."
    );
    describe_counter!(
        "summarizer_summaries_written_total",
        "Summarizer write attempts by target level and result."
    );
    describe_histogram!(
        "summarizer_duration_seconds",
        "Summarizer pass latency in seconds."
    );
    describe_counter!(
        "context_pack_requests_total",
        "Context pack build requests."
    );
    describe_counter!("context_pack_cache_hits_total", "Context pack cache hits.");
    describe_counter!(
        "context_pack_cache_misses_total",
        "Context pack cache misses."
    );
    describe_counter!(
        "context_cache_replacements_total",
        "Older context cache versions replaced after refresh."
    );
    describe_histogram!(
        "context_pack_build_duration_seconds",
        "Context pack build latency in seconds."
    );
    describe_histogram!(
        "context_pack_tokens_estimate",
        "Estimated context pack token count."
    );
    describe_counter!(
        "context_pack_items_injected_total",
        "Context pack items injected by memory layer."
    );
    describe_counter!("retrieval_hits_total", "Retrieval hits by source.");
    describe_counter!(
        "inference_tokens_total",
        "Inference tokens by kind and model."
    );
    describe_counter!(
        "vllm_prefix_cache_tokens_total",
        "vLLM prefix-cache query and hit token deltas observed around orchestrated requests."
    );
    describe_counter!(
        "vllm_prompt_tokens_by_cache_source_total",
        "vLLM prompt token deltas by backend cache source observed around orchestrated requests."
    );
    describe_counter!(
        "context_cache_stale_invalidations_total",
        "Context cache invalidations triggered by new memory."
    );
    describe_counter!("memory_promotions_total", "Memory promotion decisions.");
    describe_gauge!(
        "memory_source_coverage",
        "Fraction of promoted memory with sources."
    );
}
