use anyhow::Result;
use model2vec_rs::model::StaticModel;

const EMBEDDING_MODEL_ID: &str = "minishlab/potion-base-32M";

pub const EMBEDDING_DIMENSION: usize = 512; 

pub struct EmbeddingEngine {
    model: StaticModel,
}

impl EmbeddingEngine {
    pub fn new() -> Result<Self> {
        // TODO: Consider if hf_token, normalize_embeddings, or subfolder are needed.
        // For now, using defaults as per the user's example.
        let model = StaticModel::from_pretrained(EMBEDDING_MODEL_ID, None, None, None)?;
        Ok(Self { model })
    }

    pub fn dimension(&self) -> usize {
        // Ideally, model2vec_rs would expose a way to get this from the loaded model's config.
        // For now, we hardcode it based on known model specs.
        EMBEDDING_DIMENSION
    }

    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Using default batch_size and max_length from model2vec-rs example.
        // Consider making these configurable if needed.
        let embeddings = self.model.encode(texts);
        // model.encode returns Vec<Vec<f32>>, no explicit Result needed here unless it changes.
        Ok(embeddings)
    }

    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.model.encode(&[text.to_string()]);
        embeddings.into_iter().next().ok_or_else(|| {
            anyhow::anyhow!("Failed to generate embedding for single text: {}", text)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // This test downloads a model and might be slow/network-dependent
    fn test_embedding_engine_init_and_embed() -> Result<()> {
        let engine = EmbeddingEngine::new()?;
        assert_eq!(engine.dimension(), EMBEDDING_DIMENSION);

        let sentences = vec![
            "Hello world".to_string(),
            "Rust is awesome".to_string(),
        ];
        let embeddings = engine.embed(&sentences)?;
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), EMBEDDING_DIMENSION);
        assert_eq!(embeddings[1].len(), EMBEDDING_DIMENSION);

        let single_embedding = engine.embed_one("Test sentence")?;
        assert_eq!(single_embedding.len(), EMBEDDING_DIMENSION);
        Ok(())
    }
}
