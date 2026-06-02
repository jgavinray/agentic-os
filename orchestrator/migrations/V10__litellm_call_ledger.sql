CREATE TABLE IF NOT EXISTS litellm_call_ledger (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_event_id UUID,
    trajectory_id UUID,
    context_pack_id UUID,
    namespace TEXT,
    repo TEXT,
    task TEXT,
    endpoint TEXT,
    requested_model TEXT,
    routed_model TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
