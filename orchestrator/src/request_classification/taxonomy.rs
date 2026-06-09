use serde::{Deserialize, Serialize};

macro_rules! request_classification_enums {
    (
        $(
            $(#[$enum_meta:meta])*
            pub enum $name:ident {
                $($variant:ident => $label:literal),* $(,)?
            }
        )*
    ) => {
        $(
            $(#[$enum_meta])*
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
            pub enum $name {
                $($variant,)*
                #[default]
                Unknown,
            }

            impl $name {
                pub const fn as_str(self) -> &'static str {
                    match self {
                        $(Self::$variant => $label,)*
                        Self::Unknown => "unknown",
                    }
                }

                pub fn from_label(value: &str) -> Self {
                    match value {
                        $($label => Self::$variant,)*
                        "unknown" => Self::Unknown,
                        _ => Self::Unknown,
                    }
                }
            }
        )*

        pub fn enum_inventory() -> &'static [(&'static str, &'static [&'static str])] {
            &[
                $(
                    (
                        stringify!($name),
                        &[$($label,)* "unknown"],
                    ),
                )*
            ]
        }
    };
}

request_classification_enums! {
    pub enum RequestIntent {
        Explain => "explain",
        Debug => "debug",
        Implement => "implement",
        GenerateConfig => "generate_config",
        ModifyConfig => "modify_config",
        Summarize => "summarize",
        Classify => "classify",
        Search => "search",
        Plan => "plan",
        OperateTool => "operate_tool",
    }

    pub enum RequestDomain {
        Shell => "shell",
        Kubernetes => "kubernetes",
        LlmInference => "llm_inference",
        Docker => "docker",
        Networking => "networking",
        Security => "security",
        Medical => "medical",
        Legal => "legal",
        Finance => "finance",
        Generic => "generic",
    }

    pub enum RequestArtifactType {
        PlainText => "plain_text",
        Code => "code",
        Logs => "logs",
        Yaml => "yaml",
        Json => "json",
        Sql => "sql",
        Markdown => "markdown",
        Image => "image",
        File => "file",
    }

    pub enum RequestComplexity {
        L0Trivial => "l0_trivial",
        L1Simple => "l1_simple",
        L2Moderate => "l2_moderate",
        L3Complex => "l3_complex",
        L4ToolRequired => "l4_tool_required",
        L5HighRisk => "l5_high_risk",
    }

    pub enum RequestRisk {
        None => "none",
        SecretPresent => "secret_present",
        DestructiveCommand => "destructive_command",
        ExternalCurrentInfoRequired => "external_current_info_required",
        HighStakes => "high_stakes",
        PromptInjection => "prompt_injection",
        UnsafeSecurity => "unsafe_security",
    }

    pub enum RecommendedRoute {
        DeterministicTemplate => "deterministic_template",
        SmallLocalModel => "small_local_model",
        StrongLocalModel => "strong_local_model",
        WebRequired => "web_required",
        ToolRequired => "tool_required",
        AskClarification => "ask_clarification",
        RefuseOrGuardrail => "refuse_or_guardrail",
    }

    pub enum ResponseContract {
        DirectAnswer => "direct_answer",
        StructuredJson => "structured_json",
        MarkdownSummary => "markdown_summary",
        PatchRequired => "patch_required",
        ValidationRequired => "validation_required",
        ClarificationQuestion => "clarification_question",
        Refusal => "refusal",
    }
}
