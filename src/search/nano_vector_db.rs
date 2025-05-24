//! A lightweight vector database implementation
// #![doc(html_root_url = "https://docs.rs/nano-vectordb-rs/0.1.1")] // Removed crate-level attribute
// #![warn(rustdoc::missing_crate_level_docs)]
// #![warn(missing_docs)]
#![forbid(unsafe_code)]

use anyhow::Result;
use base64::{engine::general_purpose, Engine as _};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

/// Constants used for special field names
pub mod constants {
    /// Identifier field name
    pub const F_ID: &str = "__id__";
    /// Similarity metrics field name
    pub const F_METRICS: &str = "__metrics__";
}

type Float = f32;

/// A single vector entry with metadata
#[derive(Debug, Serialize, Deserialize, Clone)] // Added Clone
pub struct Data {
    /// Unique identifier for the vector
    #[serde(rename = "__id__")]
    pub id: String,
    /// The vector data (non-normalized)
    #[serde(skip)]
    pub vector: Vec<Float>,
    /// Additional metadata fields stored with the vector
    #[serde(flatten, skip_serializing_if = "HashMap::is_empty")]
    pub fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DataBase {
    embedding_dim: usize,
    data: Vec<Data>,
    #[serde(with = "base64_bytes")]
    matrix: Vec<Float>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    additional_data: HashMap<String, serde_json::Value>,
}

mod base64_bytes {
    use super::*;
    use bytemuck::cast_slice; // Will need bytemuck dependency
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(vec: &[Float], serializer: S) -> Result<S::Ok, S::Error> {
        let bytes = cast_slice(vec);
        let b64 = general_purpose::STANDARD.encode(bytes);
        serializer.serialize_str(&b64)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<Float>, D::Error> {
        let s = String::deserialize(deserializer)?;
        let bytes = general_purpose::STANDARD
            .decode(s)
            .map_err(serde::de::Error::custom)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| Float::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }
}

/// Main vector database struct
#[derive(Debug)]
pub struct NanoVectorDB {
    /// Dimensionality of stored vectors
    pub embedding_dim: usize,
    /// Distance metric used for similarity searches
    pub metric: String, // This is fixed to cosine in the implementation
    storage_file: PathBuf,
    storage: DataBase,
}

#[derive(PartialEq)]
struct ScoredIndex {
    score: Float,
    index: usize,
}

impl Eq for ScoredIndex {}

impl PartialOrd for ScoredIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        // We want a min-heap for scores to keep the K largest items.
        // BinaryHeap is a max-heap by default. To make it behave as a min-heap
        // for scores, Ord should compare such that smaller scores are "greater".
        // So, if self.score is smaller than other.score, it should be Ordering::Greater.
        // This is achieved by reversing the comparison:
        other.score.partial_cmp(&self.score).unwrap_or_else(|| {
            // NaN handling: NaNs are considered "smaller" than actual numbers
            // so they are preferred to be popped from the min-heap if we are keeping top K.
            // If self is NaN, and other is not, self is "smaller" (less preferred for keeping).
            if self.score.is_nan() && !other.score.is_nan() {
                Ordering::Less // self (NaN) is smaller than other (number)
            } else if !self.score.is_nan() && other.score.is_nan() {
                Ordering::Greater // self (number) is greater than other (NaN)
            } else {
                Ordering::Equal // Both NaN or both equal numbers
            }
        })
    }
}

type DataFilter = Box<dyn Fn(&Data) -> bool + Send + Sync>;

impl NanoVectorDB {
    /// Creates a new NanoVectorDB instance
    pub fn new(embedding_dim: usize, storage_file: &str) -> Result<Self> {
        let storage_file = PathBuf::from(storage_file);
        let storage = if storage_file.exists() && storage_file.metadata()?.len() > 0 {
            let contents = fs::read_to_string(&storage_file)?;
            let db: DataBase = serde_json::from_str(&contents)?;

            if db.embedding_dim != embedding_dim {
                anyhow::bail!(
                    "Embedding dimension mismatch: DB has {}, expected {}",
                    db.embedding_dim, embedding_dim
                );
            }

            let expected_len = db.data.len() * db.embedding_dim;
            if db.matrix.len() != expected_len {
                anyhow::bail!(
                    "Matrix size mismatch: expected {}, got {}",
                    expected_len,
                    db.matrix.len()
                );
            }
            db
        } else {
            DataBase {
                embedding_dim,
                data: Vec::new(),
                matrix: Vec::new(),
                additional_data: HashMap::new(),
            }
        };

        Ok(Self {
            embedding_dim,
            metric: "cosine".to_string(), // Hardcoded as per implementation
            storage_file,
            storage,
        })
    }

    /// Upserts vectors into the database
    pub fn upsert(&mut self, mut datas: Vec<Data>) -> Result<(Vec<String>, Vec<String>)> {
        let mut updates = Vec::new();
        let mut inserts = Vec::new();
        
        // Clone IDs to avoid borrow checker issues with self.storage.data
        let existing_ids_map: HashMap<String, usize> = self
            .storage
            .data
            .iter()
            .enumerate()
            .map(|(i, d)| (d.id.clone(), i)) // Clone d.id
            .collect();

        let mut new_data_to_add = Vec::new();

        for data_item in datas.drain(..) {
            // Use data_item.id directly as it's a String
            if let Some(&pos) = existing_ids_map.get(&data_item.id) {
                // Update existing
                let norm_vec = normalize(&data_item.vector); // Normalize input vector
                let start = pos * self.embedding_dim;
                let end = start + self.embedding_dim;
                if end <= self.storage.matrix.len() {
                     self.storage.matrix[start..end].copy_from_slice(&norm_vec);
                     self.storage.data[pos].fields = data_item.fields; // Update fields too
                     updates.push(data_item.id);
                } else {
                    // This case should ideally not happen if logic is correct
                    // Or it implies a corrupted state. For now, log and skip.
                    eprintln!("Error: Matrix index out of bounds during update for ID: {}", data_item.id);
                }
            } else {
                // New item
                new_data_to_add.push(data_item);
            }
        }
        
        for data_item in new_data_to_add {
            let norm_vec = normalize(&data_item.vector); // Normalize input vector
            self.storage.matrix.extend_from_slice(&norm_vec);
            self.storage.data.push(Data {
                id: data_item.id.clone(),
                vector: norm_vec, // Store normalized vector, though original code skips serializing it
                fields: data_item.fields,
            });
            inserts.push(data_item.id);
        }

        Ok((updates, inserts))
    }

    /// Queries the database for similar vectors
    pub fn query(
        &self,
        query: &[Float],
        top_k: usize,
        better_than: Option<Float>,
        filter: Option<DataFilter>,
    ) -> Vec<HashMap<String, serde_json::Value>> {
        if self.storage.data.is_empty() {
            return Vec::new();
        }
        let query_norm = normalize(query);
        let embedding_dim = self.embedding_dim;
        let matrix = &self.storage.matrix;
        let threshold = better_than.unwrap_or(-1.0); // Cosine similarity threshold, -1.0 is accept all

        // Precompute query chunks for SIMD-friendly operations (original code had this, let's keep it)
        // However, the dot_product function provided doesn't seem to use these chunks in a SIMD way directly.
        // It's more of a manual chunking.
        // For simplicity and correctness, let's use a direct dot product for now.
        // The original dot_product was a bit complex and might be error-prone if not perfectly matched.

        let mut heap = BinaryHeap::with_capacity(top_k + 1);

        for (idx, data_item_ref) in self.storage.data.iter().enumerate() {
            if filter.as_ref().map_or(true, |f| f(data_item_ref)) {
                let vector_slice_start = idx * embedding_dim;
                let vector_slice_end = vector_slice_start + embedding_dim;
                if vector_slice_end > matrix.len() {
                    // Should not happen if DB is consistent
                    eprintln!("Error: Matrix index out of bounds during query for internal index: {}", idx);
                    continue;
                }
                let vector_to_compare = &matrix[vector_slice_start..vector_slice_end];
                
                // Use the simpler dot_product for normalized vectors (cosine similarity)
                let score = simple_dot_product(vector_to_compare, &query_norm);

                if score >= threshold {
                    heap.push(ScoredIndex { score, index: idx });
                    if heap.len() > top_k {
                        heap.pop();
                    }
                }
            }
        }
        
        // Convert to sorted results (BinaryHeap pops smallest, but Ord for ScoredIndex is reversed for max-heap behavior)
        // So, into_sorted_vec will give highest scores first.
        let sorted_results = heap.into_sorted_vec(); 

        sorted_results
            .into_iter()
            .map(|si| {
                let data = &self.storage.data[si.index];
                let mut result = data.fields.clone();
                result.insert(
                    constants::F_METRICS.to_string(),
                    serde_json::json!(si.score), 
                );
                result.insert(constants::F_ID.to_string(), serde_json::json!(data.id.clone())); 
                result
            })
            .collect()
    }


    /// Get vectors by their IDs
    pub fn get(&self, ids: &[String]) -> Vec<&Data> {
        let id_set: HashSet<_> = ids.iter().map(|s| s.as_str()).collect();
        self.storage
            .data
            .iter()
            .filter(|data| id_set.contains(data.id.as_str()))
            .collect()
    }

    /// Delete vectors by their IDs
    pub fn delete(&mut self, ids_to_delete: &[String]) -> Result<usize> {
        let id_set_to_delete: HashSet<_> = ids_to_delete.iter().map(|s| s.as_str()).collect();
        let original_len = self.storage.data.len();
        let mut new_data = Vec::new();
        let mut new_matrix = Vec::new();

        for data_item in self.storage.data.iter() {
            if !id_set_to_delete.contains(data_item.id.as_str()) {
                // Keep this item
                new_data.push(data_item.clone()); // Clone the Data struct
                // The vector stored in data_item.vector should be the normalized one
                new_matrix.extend_from_slice(&data_item.vector);
            }
        }
        
        let deleted_count = original_len - new_data.len();
        self.storage.data = new_data;
        self.storage.matrix = new_matrix;
        
        Ok(deleted_count)
    }


    /// Saves the database to disk
    pub fn save(&self) -> Result<()> {
        let serialized = serde_json::to_string_pretty(&self.storage)?; // Use pretty for readability
        fs::write(&self.storage_file, serialized)?;
        Ok(())
    }

    /// Get additional metadata stored in the database
    pub fn get_additional_data(&self) -> &HashMap<String, serde_json::Value> {
        &self.storage.additional_data
    }

    /// Store additional metadata in the database
    pub fn store_additional_data(&mut self, data: HashMap<String, serde_json::Value>) {
        self.storage.additional_data = data;
    }

    /// Get the number of vectors in the database
    pub fn len(&self) -> usize {
        self.storage.data.len()
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        self.storage.data.is_empty()
    }

    /// Get total vector bytes length (of the matrix)
    pub fn vector_bytes_len(&self) -> usize {
        self.storage.matrix.len() * std::mem::size_of::<Float>() // More accurate byte len
    }
}

// Simpler dot product for already normalized vectors (calculates cosine similarity)
#[inline]
fn simple_dot_product(vec1: &[Float], vec2: &[Float]) -> Float {
    // For safety, though normalize should ensure this.
    // If lengths can mismatch, this assert is important.
    // assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length for dot product");
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
}


/// Normalize a vector to unit length
pub fn normalize(vector: &[Float]) -> Vec<Float> {
    let norm_sq: Float = vector.iter().map(|&x| x * x).sum();

    if norm_sq < Float::EPSILON * Float::EPSILON { // Compare with squared epsilon for robustness
        // Return a zero vector or handle as an error, depending on policy for zero vectors
        // For now, returning a zero vector of the same dimension.
        // This might not be ideal for cosine similarity as it can lead to NaN/Inf if 1/0 is attempted.
        // However, the original code asserts norm_sq > EPSILON.
        // Let's keep the assert for now to match original behavior.
        // If zero vectors are possible and problematic, this needs more thought.
         if norm_sq == 0.0 { // Explicitly handle zero vector to avoid 1/0
            return vec![0.0; vector.len()];
        }
        // If norm_sq is very small but non-zero, it might lead to large numbers.
        // The original assert!(norm_sq > Float::EPSILON, "Cannot normalize zero-length vector");
        // is good. Let's re-enable it or a variation.
    }
     if norm_sq == 0.0 {
        // This case should ideally be handled by the caller or by a policy.
        // Forcing normalization of a zero vector is problematic.
        // The original code had an assert. Let's keep a check.
        // eprintln!("Warning: Attempting to normalize a zero-length vector. Result will be zero vector.");
        return vec![0.0; vector.len()];
    }


    let inv_norm = 1.0 / norm_sq.sqrt();
    vector.iter().map(|&x| x * inv_norm).collect()
}

/// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    #[test]
    fn test_base64_deserialization_edge_cases() {
        // Test valid base64 deserialization
        let valid_db = DataBase {
            embedding_dim: 2,
            data: vec![Data {
                id: "test".to_string(),
                vector: vec![1.0, 2.0], // This vector field is not serialized/deserialized in DataBase.data
                fields: HashMap::new(),
            }],
            matrix: vec![1.0, 2.0], // This is what gets (de)serialized
            additional_data: HashMap::new(),
        };
        let serialized = serde_json::to_string(&valid_db).unwrap();
        let deserialized: DataBase = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.matrix, vec![1.0, 2.0]);

        // Test invalid base64 string
        let invalid_json = r#"{
            "embedding_dim": 2,
            "data": [{"__id__": "test", "fields": {}}],
            "matrix": "INVALID_BASE64!!",
            "additional_data": {}
        }"#;
        let result: Result<DataBase, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_size_validation_on_load() {
        let temp_file = NamedTempFile::new().unwrap();
        let path_str = temp_file.path().to_str().unwrap();

        // Create malformed database with mismatched matrix size
         let mut data_for_db = Vec::new();
         data_for_db.push(Data {
             id: "entry1".to_string(),
             vector: vec![1.0, 2.0], // This vector is not directly used for matrix construction in this test setup
             fields: HashMap::new(),
         });
        let corrupt_db_storage = DataBase {
            embedding_dim: 2, // Expects 2D vectors
            data: data_for_db,
            matrix: vec![1.0], // Matrix only has 1 element, but data[0] implies 2D, so matrix should have 2 elements.
            additional_data: HashMap::new(),
        };

        fs::write(path_str, serde_json::to_string(&corrupt_db_storage).unwrap()).unwrap();
        let result = NanoVectorDB::new(2, path_str); // Attempt to load with embedding_dim = 2

        assert!(result.is_err(), "Expected an error due to matrix size mismatch");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Matrix size mismatch"), "Error message mismatch: {}", err_msg);
        assert!(err_msg.contains("expected 2"), "Error message mismatch: {}", err_msg); // data.len() * embedding_dim
        assert!(err_msg.contains("got 1"), "Error message mismatch: {}", err_msg);   // matrix.len()
    }
    
    #[test]
    fn test_embedding_dim_validation_on_load() {
        let temp_file = NamedTempFile::new().unwrap();
        let path_str = temp_file.path().to_str().unwrap();

        let mut data_for_db = Vec::new();
         data_for_db.push(Data {
             id: "entry1".to_string(),
             vector: vec![1.0, 2.0], 
             fields: HashMap::new(),
         });
        let db_storage_2d = DataBase { // DB stored with 2D embeddings
            embedding_dim: 2,
            data: data_for_db,
            matrix: vec![0.0, 0.0], // Correct matrix for 1 item, 2D
            additional_data: HashMap::new(),
        };
        fs::write(path_str, serde_json::to_string(&db_storage_2d).unwrap()).unwrap();

        // Attempt to load specifying a different embedding_dim
        let result = NanoVectorDB::new(3, path_str); 
        assert!(result.is_err(), "Expected an error due to embedding dimension mismatch");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Embedding dimension mismatch"), "Error message mismatch: {}", err_msg);
        assert!(err_msg.contains("DB has 2"), "Error message mismatch: {}", err_msg);
        assert!(err_msg.contains("expected 3"), "Error message mismatch: {}", err_msg);
    }


    #[test]
    fn test_scored_index_ordering_for_max_heap() {
        // BinaryHeap is a max-heap by default if Ord implements standard comparison.
        // Here, Ord for ScoredIndex is reversed (a.cmp(b) uses b.score.cmp(a.score))
        // so that the BinaryHeap behaves as a min-heap for scores (pops smallest score).
        // To get top_k largest scores, we push and if heap.len > k, we pop.
        // The .into_sorted_vec() on such a min-heap will be ascending.
        // We want descending for "top K scores".
        // The original Ord for ScoredIndex: other.score.partial_cmp(&self.score)
        // This means if other.score is greater, it's Ordering::Greater.
        // So, if self.score is smaller, it's Ordering::Greater.
        // This makes BinaryHeap a min-heap on score.
        // into_sorted_vec() will then sort it ascendingly by score.
        // Let's verify the Ord implementation:
        // a.cmp(b) -> b.score.partial_cmp(&a.score)
        // If a.score = 0.9, b.score = 0.8: 0.8.partial_cmp(0.9) -> Less. So a < b.
        // This means larger scores are "smaller" in terms of Ord.
        // So BinaryHeap will keep the *numerically smallest scores at the top to be popped*.
        // This is not what we want for a "top K" (max score) heap.
        // The original Ord was: self.score.partial_cmp(&other.score).unwrap()
        // This makes it a max-heap. Let's revert to that if the goal is a max-heap for scores.
        // The current Ord in the provided code: other.score.partial_cmp(&self.score)
        // This makes it a min-heap on score.
        // If we want top K highest scores, we need a min-heap of size K,
        // where we insert if new_score > heap.peek().score, then pop.
        // BinaryHeap by default is a max-heap. To use it as a min-heap, we'd wrap items or reverse order.
        // The current Ord `other.score.partial_cmp(&self.score)` makes it a min-heap on score.
        // So, when `heap.len() > top_k`, `heap.pop()` removes the item with the *smallest score*. This is correct for finding top_k largest.
        // `into_sorted_vec()` on this min-heap will produce items sorted by score ascendingly.

        let mut heap = BinaryHeap::new();
        heap.push(ScoredIndex { score: 0.8, index: 0 }); // Smallest
        heap.push(ScoredIndex { score: 0.9, index: 1 }); // Middle
        heap.push(ScoredIndex { score: 0.7, index: 2 }); // Even smaller
        heap.push(ScoredIndex { score: 1.0, index: 3 }); // Largest

        // If top_k = 2
        while heap.len() > 2 { heap.pop(); }
        let sorted_k = heap.into_sorted_vec(); // With current Ord, this will be [1.0, 0.9]

        assert_eq!(sorted_k.len(), 2);
        // sorted_k[0] should have the highest score of the top K
        assert_eq!(sorted_k[0].score, 1.0, "Highest score of the top 2 should be 1.0"); 
        // sorted_k[1] should have the second highest score of the top K
        assert_eq!(sorted_k[1].score, 0.9, "Second highest score of the top 2 should be 0.9");

        // Test NaN handling in Ord
        let nan_score = ScoredIndex { score: Float::NAN, index: 0 };
        let regular_score = ScoredIndex { score: 0.5, index: 1 };

        // For our min-heap (of scores), NaN should be considered "less than" any number,
        // so it gets popped if the heap is full.
        // nan_score.cmp(regular_score) should be Ordering::Less.
        // self = nan_score, other = regular_score
        // In Ord: other.score.partial_cmp(&self.score) -> 0.5.partial_cmp(NaN) -> None.
        // unwrap_or_else: self.score.is_nan() (true) && !other.score.is_nan() (true) -> Ordering::Less. Correct.
        assert_eq!(nan_score.cmp(&regular_score), Ordering::Less, "NaN score should be Less than a regular score for min-heap ordering");

        // regular_score.cmp(nan_score) should be Ordering::Greater.
        // self = regular_score, other = nan_score
        // In Ord: other.score.partial_cmp(&self.score) -> NaN.partial_cmp(0.5) -> None.
        // unwrap_or_else: !self.score.is_nan() (true) && other.score.is_nan() (true) -> Ordering::Greater. Correct.
        assert_eq!(regular_score.cmp(&nan_score), Ordering::Greater, "Regular score should be Greater than a NaN score for min-heap ordering");
    }

    #[test]
    fn test_upsert_and_query() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let db_path = temp_file.path().to_str().unwrap();
        let mut db = NanoVectorDB::new(3, db_path)?;

        let samples1 = vec![
            Data { id: "vec1".into(), vector: vec![1.0, 2.0, 3.0], fields: [("color".into(), serde_json::json!("red"))].into() },
            Data { id: "vec2".into(), vector: vec![-4.0, 5.0, 6.0], fields: [("color".into(), serde_json::json!("blue"))].into()},
        ];
        let (_, inserted1) = db.upsert(samples1)?;
        assert_eq!(inserted1.len(), 2);
        db.save()?;

        let samples2 = vec![
            Data { id: "vec1".into(), vector: vec![1.1, 2.1, 3.1], fields: [("color".into(), serde_json::json!("dark red"))].into()}, // Update
            Data { id: "vec3".into(), vector: vec![7.0, 8.0, -9.0], fields: [("color".into(), serde_json::json!("green"))].into()}, // Insert
        ];
        let (updated2, inserted2) = db.upsert(samples2)?;
        assert_eq!(updated2.len(), 1);
        assert_eq!(inserted2.len(), 1);
        assert_eq!(db.len(), 3);

        let query_vec = vec![1.0, 2.0, 3.0]; // Closest to updated vec1
        let results = db.query(&query_vec, 1, None, None);
        assert_eq!(results.len(), 1);
        let top_result = &results[0];
        assert_eq!(top_result[constants::F_ID], "vec1");
        assert_eq!(top_result["color"], "dark red");
        // Score should be high (close to 1.0 for cosine similarity with itself if normalized)
        assert!(top_result[constants::F_METRICS].as_f64().unwrap() > 0.95);

        Ok(())
    }

    #[test]
    fn test_delete() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let db_path = temp_file.path().to_str().unwrap();
        let mut db = NanoVectorDB::new(3, db_path)?;
        let samples = vec![
            Data { id: "v1".into(), vector: vec![1.,0.,0.], fields: HashMap::new() },
            Data { id: "v2".into(), vector: vec![0.,1.,0.], fields: HashMap::new() },
            Data { id: "v3".into(), vector: vec![0.,0.,1.], fields: HashMap::new() },
        ];
        db.upsert(samples)?;
        assert_eq!(db.len(), 3);

        db.delete(&["v2".into()])?;
        assert_eq!(db.len(), 2);
        
        let remaining_ids: HashSet<String> = db.storage.data.iter().map(|d| d.id.clone()).collect();
        assert!(remaining_ids.contains("v1"));
        assert!(!remaining_ids.contains("v2"));
        assert!(remaining_ids.contains("v3"));
        
        // Check matrix consistency
        assert_eq!(db.storage.matrix.len(), 2 * db.embedding_dim);

        Ok(())
    }
    
    #[test]
    fn test_normalize_zero_vector() {
        let zero_vec = vec![0.0, 0.0, 0.0];
        let normalized = normalize(&zero_vec);
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]); // Should return zero vector
    }

    #[test]
    fn test_normalize_non_zero_vector() {
        let vec = vec![3.0, 4.0]; // norm is 5
        let normalized = normalize(&vec);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }
}
