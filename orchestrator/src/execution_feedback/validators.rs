#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationKind {
    Compile,
    Test,
    Lint,
    TypeCheck,
    Schema,
    StaticAnalysis,
    Other,
}

pub const VALIDATOR_TYPES: [&str; 7] = [
    "compile",
    "test",
    "lint",
    "type_check",
    "schema",
    "static_analysis",
    "other",
];

pub fn validate_validator_type(value: Option<&str>) -> Result<(), String> {
    if let Some(value) = value {
        if !VALIDATOR_TYPES.contains(&value) {
            return Err(format!("invalid validator_type `{value}`"));
        }
    }
    Ok(())
}

pub(crate) fn validator_type_str(kind: ValidationKind) -> &'static str {
    match kind {
        ValidationKind::Compile => "compile",
        ValidationKind::Test => "test",
        ValidationKind::Lint => "lint",
        ValidationKind::TypeCheck => "type_check",
        ValidationKind::Schema => "schema",
        ValidationKind::StaticAnalysis => "static_analysis",
        ValidationKind::Other => "other",
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorSpec {
    pub validator: &'static str,
    pub kind: ValidationKind,
}

pub fn classify_validator(tool_name: &str, content: &str) -> Option<ValidatorSpec> {
    // Validator names are intentionally bounded for metrics cardinality. Match
    // by substring because tool names often arrive as full shell commands.
    let name = tool_name.to_ascii_lowercase();
    let body = content.to_ascii_lowercase();
    if name.contains("pytest") {
        return Some(ValidatorSpec {
            validator: "pytest",
            kind: ValidationKind::Test,
        });
    }
    if name.contains("npm test") || name == "npm-test" || name.contains("jest") {
        return Some(ValidatorSpec {
            validator: "npm test",
            kind: ValidationKind::Test,
        });
    }
    if name.contains("cargo") {
        let kind = if name.contains("test") || body.contains("test result:") {
            ValidationKind::Test
        } else if name.contains("clippy") {
            ValidationKind::Lint
        } else {
            ValidationKind::Compile
        };
        return Some(ValidatorSpec {
            validator: "cargo",
            kind,
        });
    }
    if name.contains("eslint") {
        return Some(ValidatorSpec {
            validator: "eslint",
            kind: ValidationKind::Lint,
        });
    }
    if name.contains("tsc") {
        return Some(ValidatorSpec {
            validator: "tsc",
            kind: ValidationKind::TypeCheck,
        });
    }
    if name.contains("mypy") {
        return Some(ValidatorSpec {
            validator: "mypy",
            kind: ValidationKind::TypeCheck,
        });
    }
    if name.contains("ruff") {
        return Some(ValidatorSpec {
            validator: "ruff",
            kind: ValidationKind::Lint,
        });
    }
    if name.contains("terraform") {
        return Some(ValidatorSpec {
            validator: "terraform",
            kind: ValidationKind::Schema,
        });
    }
    if name.contains("kubectl") {
        return Some(ValidatorSpec {
            validator: "kubectl",
            kind: ValidationKind::Schema,
        });
    }
    None
}
