use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct MemoryNote {
    pub id: String,
    pub date: String,
    pub title: Option<String>,
    pub content: String,
    pub updated_at: i64,
    pub archived: bool,
}

#[derive(Debug, Deserialize)]
struct NotesResponse {
    notes: Vec<MemoryNote>,
}

pub async fn recent_notes(
    http: &reqwest::Client,
    base_url: &str,
    days: usize,
    limit: usize,
) -> Result<Vec<MemoryNote>, anyhow::Error> {
    let base_url = base_url.trim_end_matches('/');
    let url = format!("{base_url}/api/recent");
    let response = http
        .get(url)
        .query(&[
            ("days", days.to_string()),
            ("limit", limit.to_string()),
            ("include_archived", "false".to_string()),
        ])
        .send()
        .await?;
    if !response.status().is_success() {
        anyhow::bail!("Total Recall recent notes returned {}", response.status());
    }
    Ok(response.json::<NotesResponse>().await?.notes)
}

pub async fn search_notes(
    http: &reqwest::Client,
    base_url: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<MemoryNote>, anyhow::Error> {
    let base_url = base_url.trim_end_matches('/');
    let url = format!("{base_url}/api/search");
    let response = http
        .post(url)
        .json(&serde_json::json!({
            "query": query,
            "limit": limit,
            "include_archived": false,
        }))
        .send()
        .await?;
    if !response.status().is_success() {
        anyhow::bail!("Total Recall search notes returned {}", response.status());
    }
    Ok(response.json::<NotesResponse>().await?.notes)
}
