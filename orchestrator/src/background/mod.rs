pub mod trajectory;

use std::future::Future;

use crate::state::AppState;

pub(crate) fn spawn_bounded_background<F>(state: &AppState, job: &'static str, fut: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    let gate = state.background_work.clone();
    tokio::spawn(async move {
        let _permit = match gate.acquire_owned().await {
            Ok(permit) => permit,
            Err(e) => {
                tracing::warn!(job, "background work gate closed: {e}");
                return;
            }
        };
        fut.await;
    });
}
