use anyhow::{Result, Context};
use std::collections::HashMap; // For NanoDBData fields
use crate::search::nano_vector_db::{NanoVectorDB, Data as NanoDBData, constants as NanoDBConstants};
use crate::search::embedding_engine::EMBEDDING_DIMENSION; // To ensure consistency if needed, or use passed dimension

const DB_PATH: &str = "ann_engine_nanodb.json"; // Path for the NanoVectorDB file

// ANN_METRIC is not directly used by NanoVectorDB as it's fixed to cosine,
// but we keep the constant here if other parts of the code might refer to it conceptually.
// pub const ANN_METRIC: Metric = Metric::CosineSimilarity; // Hora specific, can be removed.

pub struct AnnEngine {
    db: NanoVectorDB,
    dimension: usize, // Store dimension for validation if needed, NanoDB also stores it
}

impl AnnEngine {
    pub fn new(dimension: usize) -> Result<Self> {
        let db = NanoVectorDB::new(dimension, DB_PATH)
            .with_context(|| format!("Failed to initialize NanoVectorDB for AnnEngine at path: {}", DB_PATH))?;
        Ok(Self { db, dimension })
    }

    pub fn add_items_batch(&mut self, embeddings: &[Vec<f32>], ids: &[String]) -> Result<()> {
        if embeddings.len() != ids.len() {
            return Err(anyhow::anyhow!(
                "Embeddings and IDs count mismatch: {} vs {}",
                embeddings.len(),
                ids.len()
            ));
        }

        let mut nano_data_items: Vec<NanoDBData> = Vec::with_capacity(embeddings.len());

        for (embedding, id_str) in embeddings.iter().zip(ids.iter()) {
            if embedding.len() != self.dimension {
                return Err(anyhow::anyhow!(
                    "Embedding dimension mismatch for item '{}'. Expected {}, got {}.",
                    id_str,
                    self.dimension,
                    embedding.len()
                ));
            }
            // The `ids` provided by NutritionalIndex are stringified usize indices ("0", "1", ...).
            // These will be the `id` field in NanoDBData.
            // We don't need an extra payload field if the ID itself is the index.
            let data_item = NanoDBData {
                id: id_str.clone(),
                vector: embedding.clone(),
                fields: HashMap::new(), // No extra fields needed for now, ID is the key.
            };
            nano_data_items.push(data_item);
        }

        if !nano_data_items.is_empty() {
            self.db.upsert(nano_data_items)
                .with_context(|| "Failed to upsert batch to NanoVectorDB")?;
            self.db.save()
                .with_context(|| "Failed to save NanoVectorDB after batch upsert")?;
        }
        Ok(())
    }

    // This method is now a no-op as NanoVectorDB doesn't have a separate build step.
    // It's kept for API compatibility with NutritionalIndex.
    pub fn build_index(&mut self) -> Result<()> {
        // println!("AnnEngine: build_index() called (no-op for NanoVectorDB).");
        Ok(())
    }

    pub fn search(&self, query_embedding: &[f32], k: usize) -> Vec<String> {
        if query_embedding.len() != self.dimension {
            eprintln!(
                "Search query embedding dimension mismatch. Expected {}, got {}.",
                self.dimension,
                query_embedding.len()
            );
            return Vec::new();
        }

        let search_results_maps = self.db.query(query_embedding, k, None, None);
        
        search_results_maps
            .into_iter()
            .filter_map(|result_map| {
                match result_map.get(NanoDBConstants::F_ID) {
                    Some(id_val) => id_val.as_str().map(String::from),
                    None => {
                        eprintln!("Search result from NanoVectorDB missing ID field.");
                        None
                    }
                }
            })
            .collect()
    }

    pub fn item_count(&self) -> usize {
        self.db.len()
    }

    // Helper to clean up the DB file, useful for tests
    #[cfg(test)]
    fn cleanup_db_file() -> Result<()> {
        let path = std::path::Path::new(DB_PATH);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // EMBEDDING_DIMENSION is already imported if this module is part of the crate
    // use crate::search::embedding_engine::EMBEDDING_DIMENSION; 
    use rand::Rng; // For generating dummy embeddings

    fn generate_dummy_embeddings(count: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<String>) {
        let mut rng = rand::thread_rng();
        let embeddings = (0..count)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>())
            .collect();
        let ids = (0..count).map(|i| format!("{}", i)).collect(); // IDs are "0", "1", ...
        (embeddings, ids)
    }

    #[test]
    fn test_ann_engine_new_add_search() -> Result<()> {
        AnnEngine::cleanup_db_file()?; // Clean up before test

        let dim = EMBEDDING_DIMENSION; // Use the global const
        let mut engine = AnnEngine::new(dim)?;

        let (embeddings, ids) = generate_dummy_embeddings(100, dim);
        engine.add_items_batch(&embeddings, &ids)?;
        assert_eq!(engine.item_count(), 100);

        // build_index is a no-op, but we can call it to ensure it doesn't error
        engine.build_index()?;

        let query_embedding = embeddings[0].clone(); // Query with the first embedding
        let results = engine.search(&query_embedding, 5);
        
        assert!(!results.is_empty(), "Search returned no results");
        assert_eq!(results.len(), 5.min(engine.item_count()), "Search returned incorrect number of results");
        
        // The closest item to embeddings[0] should be "0" (its own ID)
        assert_eq!(results[0], "0", "The first result should be the item itself");

        AnnEngine::cleanup_db_file()?; // Clean up after test
        Ok(())
    }

    #[test]
    fn test_ann_engine_persistence() -> Result<()> {
        AnnEngine::cleanup_db_file()?;
        let dim = EMBEDDING_DIMENSION;

        // Create engine, add items, it saves automatically
        let mut engine1 = AnnEngine::new(dim)?;
        let (embeddings, ids) = generate_dummy_embeddings(10, dim);
        engine1.add_items_batch(&embeddings, &ids)?;
        assert_eq!(engine1.item_count(), 10);
        
        // Drop engine1, then create a new one (engine2) which should load from DB_PATH
        drop(engine1);
        let engine2 = AnnEngine::new(dim)?;
        assert_eq!(engine2.item_count(), 10, "Engine2 should load 10 items from persisted DB");

        let query_embedding = embeddings[5].clone();
        let results = engine2.search(&query_embedding, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], "5");

        AnnEngine::cleanup_db_file()?;
        Ok(())
    }
}
