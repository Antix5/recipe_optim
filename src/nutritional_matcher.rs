use anyhow::{Result, Context, anyhow};
use std::path::Path;
use std::collections::HashMap;
use serde::{Serialize, Deserialize}; // Added missing serde derives

use crate::search::embedding_engine::{EmbeddingEngine, EMBEDDING_DIMENSION};
use crate::search::ann_engine::AnnEngine;
use crate::search::data_loader::load_ciqual_nutritional_data;
use crate::recipe_converter::{CiqualFoodItem, CleanedIngredient, CalculatedNutritionalInfo};
use crate::api_connection::endpoints::{
    ChatCompletionRequest, ChatMessage, JsonSchema, JsonSchemaDefinition, JsonSchemaProperty,
    ResponseFormat, Provider,
};
// ApiConnectionError is not directly used, but might be relevant if we add more specific error handling
// use crate::api_connection::connection::ApiConnectionError; 

// Struct for Qwen's response for disambiguation
#[derive(Debug, Serialize, Deserialize, Clone)]
struct DisambiguationResponse {
    best_match_index: i32, // 0 for no match, 1-K for candidate index
}

fn get_disambiguation_json_schema(candidate_count: usize) -> JsonSchemaDefinition {
    let mut properties_map = HashMap::new();
    properties_map.insert(
        "best_match_index".to_string(),
        JsonSchemaProperty {
            property_type: "integer".to_string(),
            description: Some(format!(
                "The 1-based index of the best matching candidate (1 to {}). Respond with 0 if no candidate is a good match.",
                candidate_count
            )),
            r#enum: None, 
            items: None,
        },
    );

    JsonSchemaDefinition {
        name: "disambiguation_schema".to_string(),
        strict: Some(true),
        schema: JsonSchema {
            schema_type: "object".to_string(),
            properties: Some(properties_map),
            required: Some(vec!["best_match_index".to_string()]),
            additional_properties: Some(false),
        },
    }
}

pub struct NutritionalIndex {
    embedding_engine: EmbeddingEngine,
    ann_engine: AnnEngine,
    ciqual_data: Vec<CiqualFoodItem>, // Stores all loaded Ciqual items
}

impl NutritionalIndex {
    pub fn new(ciqual_csv_path: &Path, _api_key_env_var: &str) -> Result<Self> {
        println!("Initializing NutritionalIndex...");
        println!(" > Loading Ciqual nutritional data from {:?}...", ciqual_csv_path);
        let ciqual_data = load_ciqual_nutritional_data(ciqual_csv_path)
            .with_context(|| format!("Failed to load Ciqual data from {:?}", ciqual_csv_path))?;
        println!(" > Ciqual data loaded: {} items.", ciqual_data.len());

        println!(" > Initializing embedding engine...");
        let embedding_engine = EmbeddingEngine::new()
            .with_context(|| "Failed to initialize embedding engine")?;
        
        let food_names: Vec<String> = ciqual_data.iter().map(|item| item.name.clone()).collect();
        println!(" > Generating embeddings for {} Ciqual food names...", food_names.len());
        let embeddings = embedding_engine.embed(&food_names)
            .with_context(|| "Failed to generate embeddings for Ciqual food names")?;
        println!(" > Embeddings generated. Count: {}", embeddings.len());

        if embeddings.is_empty() {
            return Err(anyhow::anyhow!("No embeddings were generated for Ciqual food names."));
        }
        println!(" > Inspecting generated embeddings (first few and overall checks)...");
        for (i, emb) in embeddings.iter().enumerate().take(3) { 
            println!("   - Embedding {} (first 5 dims): {:?}", i, emb.iter().take(5).collect::<Vec<_>>());
        }

        let mut found_nan_inf = false;
        let mut found_zero_vector = false;
        let mut found_wrong_dimension = false;

        for (idx, emb) in embeddings.iter().enumerate() {
            if emb.len() != EMBEDDING_DIMENSION {
                eprintln!("[ERROR] Embedding at index {} has incorrect dimension: {}. Expected: {}", idx, emb.len(), EMBEDDING_DIMENSION);
                found_wrong_dimension = true;
            }
            if emb.iter().any(|val| val.is_nan() || val.is_infinite()) {
                eprintln!("[ERROR] Embedding at index {} contains NaN or Infinity.", idx);
                found_nan_inf = true;
            }
            if emb.iter().all(|&val| val == 0.0) {
                eprintln!("[WARNING] Embedding at index {} is an all-zero vector.", idx);
                found_zero_vector = true; 
            }
        }

        if found_wrong_dimension {
            return Err(anyhow::anyhow!("One or more embeddings had an incorrect dimension. Cannot proceed."));
        }
        if found_nan_inf {
            return Err(anyhow::anyhow!("One or more embeddings contained NaN or Infinity. Cannot proceed."));
        }
        if found_zero_vector {
            println!("[INFO] Found one or more all-zero vectors. This might affect ANN performance or stability.");
        }
        
        let mut unique_embeddings = std::collections::HashSet::new();
        let mut duplicate_count = 0;
        for emb in embeddings.iter() {
            let emb_str = emb.iter().map(|f| f.to_string()).collect::<Vec<String>>().join(",");
            if !unique_embeddings.insert(emb_str) {
                duplicate_count += 1;
            }
        }
        if duplicate_count > 0 {
            println!("[WARNING] Found {} duplicate embeddings out of {}. This might impact HNSW construction.", duplicate_count, embeddings.len());
        }
        println!(" > Embedding inspection complete.");

        println!(" > Initializing ANN engine with dimension {}...", EMBEDDING_DIMENSION);
        let mut ann_engine = AnnEngine::new(EMBEDDING_DIMENSION)
            .with_context(|| "Failed to initialize AnnEngine")?; 
        
        let string_ann_ids: Vec<String> = (0..embeddings.len()).map(|i| i.to_string()).collect();

        println!(" > Adding {} embeddings to ANN engine with sequential IDs (0 to {})...", embeddings.len(), embeddings.len().saturating_sub(1));
        ann_engine.add_items_batch(&embeddings, &string_ann_ids)
             .with_context(|| "Failed to add Ciqual embeddings to ANN engine")?;
        
        println!(" > Building ANN index (no-op for NanoVectorDB)...");
        ann_engine.build_index().with_context(|| "Failed to build ANN index (should be no-op)")?;
        println!(" > ANN items processed. Item count: {}", ann_engine.item_count());

        println!("NutritionalIndex initialized successfully.");
        Ok(Self {
            embedding_engine,
            ann_engine, 
            ciqual_data,
        })
    }

    pub async fn find_and_calculate_nutrition(
        &self,
        ingredient: &CleanedIngredient,
        api_key_env_var: &str, 
        progress_updater: &impl Fn(String),
    ) -> Result<Option<CalculatedNutritionalInfo>> {
        progress_updater(format!("   -> Matching ingredient: '{}'", ingredient.ingredient_name));

        let query_embedding = self.embedding_engine.embed_one(&ingredient.ingredient_name)
            .with_context(|| format!("Failed to generate embedding for recipe ingredient: {}", ingredient.ingredient_name))?;

        let k = 10; 
        let ann_search_results_str_ids: Vec<String> = self.ann_engine.search(&query_embedding, k);
        
        let candidate_vec_indices: Vec<usize> = ann_search_results_str_ids.iter()
            .filter_map(|s_id| s_id.parse::<usize>().ok())
            .collect();

        if candidate_vec_indices.is_empty() {
            progress_updater(format!("   -> No ANN candidates found for '{}'.", ingredient.ingredient_name));
            return Ok(None);
        }

        let candidates: Vec<&CiqualFoodItem> = candidate_vec_indices.iter()
            .filter_map(|&vec_idx| self.ciqual_data.get(vec_idx)) 
            .collect();
        
        if candidates.is_empty() {
            progress_updater(format!("   -> ANN candidate indices did not map to Ciqual items for '{}'. Indices: {:?}", ingredient.ingredient_name, candidate_vec_indices));
            return Ok(None);
        }

        progress_updater(format!("   -> Top {} ANN candidates for '{}':", candidates.len(), ingredient.ingredient_name));
        let mut candidate_prompt_list = String::new();
        for (i, candidate_item) in candidates.iter().enumerate() {
            let line = format!("{}. \"{}\"", i + 1, candidate_item.name);
            progress_updater(format!("     {}", line));
            candidate_prompt_list.push_str(&line);
            candidate_prompt_list.push('\n');
        }

        let disambiguation_system_prompt = "/no_thinking
You are a food item matching assistant. Your task is to choose the best match for a given recipe ingredient from a list of candidate food items from a nutritional database.
Consider the ingredient name and any preparation notes.
**Crucially, pay close attention to the form of the user's ingredient (e.g., if it's a 'flour', a 'powder', a 'whole raw' item, a 'cooked' item, a 'liquid', 'puree', etc.) and strongly prefer CIQUAL candidates that match this specific form.**
For example, if the user ingredient is 'wheat flour', prefer candidates like 'Wheat flour, type X' over 'Wheat, whole, raw'. If the user ingredient is 'apple puree', prefer 'Fruits puree, apple' over 'Apple, raw'.
If the user ingredient mentions a specific state like 'cooked' or 'raw', try to match that state.

Respond ONLY with a JSON object strictly adhering to the provided schema: { \"best_match_index\": number }
The number should be the 1-based index of the chosen candidate. 
If none of the candidates are a good match, or if the best apparent match is still significantly different in form or type despite your best effort to match form, respond with 0.";

        let disambiguation_user_prompt = format!(
"Recipe Ingredient: \"{}\"
Preparation Notes: \"{}\"

Candidate Nutritional Database Items:
{}
Which candidate item (by number, 1 to {}) is the best semantic and form-based match for the recipe ingredient?
If none are a good match, respond with 0.",
            ingredient.ingredient_name,
            ingredient.preparation_notes,
            candidate_prompt_list.trim(),
            candidates.len()
        );

        let provider = Provider::openrouter(api_key_env_var);
        let request = ChatCompletionRequest {
            model: "qwen/qwen3-32b".to_string(), 
            messages: vec![
                ChatMessage { role: "system".to_string(), content: disambiguation_system_prompt.to_string() },
                ChatMessage { role: "user".to_string(), content: disambiguation_user_prompt },
            ],
            response_format: Some(ResponseFormat {
                format_type: "json_schema".to_string(), // Corrected from "json_object" to "json_schema" if schema is provided
                json_schema: Some(get_disambiguation_json_schema(candidates.len())),
            }),
            temperature: Some(0.0), // Changed from 0.1 to 0.0 for more deterministic output
            max_tokens: Some(50),
        };

        let llm_response_content = match provider.call_chat_completion(request).await {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    let mut content_str = choice.message.content.trim().to_string();
                    // Handle potential markdown code block wrapping
                    if content_str.starts_with("```json") && content_str.ends_with("```") {
                        content_str = content_str.trim_start_matches("```json").trim_end_matches("```").trim().to_string();
                    } else if content_str.starts_with("```") && content_str.ends_with("```") {
                        content_str = content_str.trim_start_matches("```").trim_end_matches("```").trim().to_string();
                    }
                    Some(content_str)
                } else {
                    progress_updater("   -> LLM returned no choice for disambiguation.".to_string());
                    None
                }
            }
            Err(e) => {
                progress_updater(format!("   -> API call for LLM disambiguation failed: {}", e));
                None
            }
        };

        if llm_response_content.is_none() {
            return Ok(None);
        }
        let llm_content = llm_response_content.unwrap();

        let chosen_ciqual_item_option: Option<&CiqualFoodItem> = match serde_json::from_str::<DisambiguationResponse>(&llm_content) {
            Ok(disamb_response) => {
                progress_updater(format!("   -> LLM chose index: {}", disamb_response.best_match_index));
                if disamb_response.best_match_index > 0 && (disamb_response.best_match_index as usize) <= candidates.len() {
                    candidates.get((disamb_response.best_match_index - 1) as usize).copied()
                } else {
                    progress_updater("   -> LLM indicated no good match or invalid index.".to_string());
                    None
                }
            }
            Err(e) => {
                progress_updater(format!("   -> Failed to parse LLM disambiguation response: {}. Raw: {}", e, llm_content));
                None
            }
        };
        
        if chosen_ciqual_item_option.is_none() {
             progress_updater(format!("   -> No definitive match found for '{}' after LLM disambiguation.", ingredient.ingredient_name));
            return Ok(None);
        }
        let chosen_ciqual_item = chosen_ciqual_item_option.unwrap();
        progress_updater(format!("   -> Matched '{}' to Ciqual item: '{}'", ingredient.ingredient_name, chosen_ciqual_item.name));

        if let Some(grams) = ingredient.quantity_grams {
            let scale = grams / 100.0;
            let calculated_info = CalculatedNutritionalInfo {
                source_ciqual_name: chosen_ciqual_item.name.clone(),
                kcal: chosen_ciqual_item.kcal_per_100g.map(|v| v * scale),
                water_g: chosen_ciqual_item.water_g_per_100g.map(|v| v * scale),
                protein_g: chosen_ciqual_item.protein_g_per_100g.map(|v| v * scale),
                carbohydrate_g: chosen_ciqual_item.carbohydrate_g_per_100g.map(|v| v * scale),
                fat_g: chosen_ciqual_item.fat_g_per_100g.map(|v| v * scale),
                sugars_g: chosen_ciqual_item.sugars_g_per_100g.map(|v| v * scale),
                fa_saturated_g: chosen_ciqual_item.fa_saturated_g_per_100g.map(|v| v * scale),
                salt_g: chosen_ciqual_item.salt_g_per_100g.map(|v| v * scale),
            };
            Ok(Some(calculated_info))
        } else {
            progress_updater(format!("   -> Cannot calculate nutrition for '{}' as quantity_grams is missing.", ingredient.ingredient_name));
            Ok(None)
        }
    }
}

// These are brought in by the `use serde::{Serialize, Deserialize};` and `use std::collections::HashMap;` at the top.
// No need to declare them again here.
// use serde::{Serialize, Deserialize};
// use std::collections::HashMap;
