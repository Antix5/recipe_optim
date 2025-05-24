pub mod ann_engine; // Restored: we will modify this existing engine
pub mod data_loader;
pub mod embedding_engine;
pub mod nano_vector_db; // Our vendored DB code

// Re-export key structs/functions if needed for easier access from outside the search module
pub use ann_engine::AnnEngine; // Restored
pub use data_loader::load_ciqual_nutritional_data;
pub use embedding_engine::EmbeddingEngine;
pub use embedding_engine::EMBEDDING_DIMENSION;
pub use nano_vector_db::{NanoVectorDB, Data as NanoDBData, constants as NanoDBConstants}; // Re-exporting from our vendored code, including constants
// pub mod vector_db_engine; // Removed - we are modifying ann_engine instead
// pub use vector_db_engine::VectorDBEngine; // Removed
