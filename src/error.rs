/// Any heap-allocated error that can cross thread boundaries.
///
/// Using a type alias here (rather than a custom enum) lets `Dataset` and
/// `Collator` implementors return any error type they like via `?` — no
/// conversion boilerplate required.
pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

/// `Result` type used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
