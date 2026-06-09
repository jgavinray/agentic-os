use crate::request_classification_types::RequestClassification;

pub(crate) struct ClassificationWriteBindings {
    pub secondary_domains: Vec<String>,
    pub risk: Vec<String>,
}

impl ClassificationWriteBindings {
    pub fn from_classification(classification: &RequestClassification) -> Self {
        Self {
            secondary_domains: classification
                .secondary_domains
                .iter()
                .map(|domain| domain.as_str().to_string())
                .collect(),
            risk: classification
                .risk
                .iter()
                .map(|risk| risk.as_str().to_string())
                .collect(),
        }
    }
}
