pub(crate) fn source_level_for_target(target_level: i32) -> Result<i32, anyhow::Error> {
    match target_level {
        1 => Ok(0),
        2 => Ok(1),
        3 => Ok(2),
        _ => anyhow::bail!("invalid summary target level: {target_level}"),
    }
}

pub(crate) fn summary_prompt_for_level(target_level: i32) -> Result<&'static str, anyhow::Error> {
    let idx = usize::try_from(target_level - 1).unwrap_or(usize::MAX);
    crate::state::SUMMARY_PROMPTS
        .get(idx)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("invalid summary target level: {target_level}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_sequence() {
        assert_eq!(source_level_for_target(1).unwrap(), 0);
        assert_eq!(source_level_for_target(2).unwrap(), 1);
        assert_eq!(source_level_for_target(3).unwrap(), 2);
        assert!(source_level_for_target(4).is_err());

        assert_ne!(
            summary_prompt_for_level(1).unwrap(),
            summary_prompt_for_level(2).unwrap()
        );
        assert_ne!(
            summary_prompt_for_level(2).unwrap(),
            summary_prompt_for_level(3).unwrap()
        );
    }
}
