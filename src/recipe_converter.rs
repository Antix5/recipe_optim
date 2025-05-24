use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;

use crate::recipe_parser::{ParsedIngredient, ParsedRecipe}; // Assuming ParsedRecipe is in recipe_parser
use crate::api_connection::endpoints::{
    ChatCompletionRequest, ChatMessage, JsonSchema, JsonSchemaDefinition, JsonSchemaProperty,
    ResponseFormat, Provider,
};
use crate::api_connection::connection::ApiConnectionError;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CleanedIngredient {
    pub raw_text: String,
    pub ingredient_name: String,
    pub original_quantity: String,
    pub original_unit: String,
    pub preparation_notes: String,
    pub quantity_grams: Option<f32>,
    pub conversion_source: String, // e.g., "LLM", "DatabaseLookup"
    pub conversion_notes: Option<String>,
    pub nutritional_info: Option<CalculatedNutritionalInfo>, // Added
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CiqualFoodItem {
    pub name: String,
    pub original_row_index: usize, // To map back if needed, or for ANN ID
    pub kcal_per_100g: Option<f32>,
    pub water_g_per_100g: Option<f32>,
    pub protein_g_per_100g: Option<f32>,
    pub carbohydrate_g_per_100g: Option<f32>,
    pub fat_g_per_100g: Option<f32>,
    pub sugars_g_per_100g: Option<f32>,
    pub fa_saturated_g_per_100g: Option<f32>,
    pub salt_g_per_100g: Option<f32>,
    // Add other fields if there are more nutritional columns from ciqual.csv
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CalculatedNutritionalInfo {
    pub source_ciqual_name: String,
    pub kcal: Option<f32>,
    pub water_g: Option<f32>,
    pub protein_g: Option<f32>,
    pub carbohydrate_g: Option<f32>,
    pub fat_g: Option<f32>,
    pub sugars_g: Option<f32>,
    pub fa_saturated_g: Option<f32>,
    pub salt_g: Option<f32>,
    // Mirror fields from CiqualFoodItem, but calculated for specific quantity
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CleanedRecipe {
    pub recipe_title: String,
    pub ingredients: Vec<CleanedIngredient>,
    pub instructions: Vec<String>,
}

// Struct for Qwen's response for gram conversion
#[derive(Debug, Serialize, Deserialize, Clone)]
struct GramConversionResponse {
    grams: Option<f32>,
    notes: String,
}

fn get_gram_conversion_json_schema() -> JsonSchemaDefinition {
    let mut properties_map = HashMap::new();
    properties_map.insert(
        "grams".to_string(),
        JsonSchemaProperty {
            property_type: "number".to_string(), 
            description: Some("The converted quantity in grams. Null if not convertible.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    properties_map.insert(
        "notes".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Any notes about the conversion, assumptions made, or errors.".to_string()),
            r#enum: None,
            items: None,
        },
    );

    JsonSchemaDefinition {
        name: "gram_conversion_schema".to_string(),
        strict: Some(true),
        schema: JsonSchema {
            schema_type: "object".to_string(),
            properties: Some(properties_map),
            required: Some(vec!["grams".to_string(), "notes".to_string()]),
            additional_properties: Some(false),
        },
    }
}

pub async fn convert_ingredients_to_grams(
    parsed_recipe: &ParsedRecipe,
    api_key_env_var: &str,
    progress_updater: impl Fn(String) + Send + Sync + 'static, 
) -> Result<CleanedRecipe, anyhow::Error> {
    let mut cleaned_ingredients: Vec<CleanedIngredient> = Vec::new();
    let provider = Provider::openrouter(api_key_env_var);

    for (index, ingredient) in parsed_recipe.ingredients.iter().enumerate() {
        progress_updater(format!(
            "Converting ingredient {}/{}: {} {} {}...",
            index + 1,
            parsed_recipe.ingredients.len(),
            ingredient.quantity,
            ingredient.unit,
            ingredient.ingredient_name
        ));

        let conversion_prompt = format!(
            "/no_thinking
You are a unit conversion assistant. Your task is to convert the given ingredient quantity to grams.
Ingredient Name: \"{}\"
Quantity: \"{}\"
Unit: \"{}\"
Preparation Notes: \"{}\"

Consider common food densities and typical weights for items specified by count (e.g., '1 large egg').
If the unit is already in grams (g), simply return that value.
If a direct conversion is impossible, highly ambiguous, or the unit is not a measure of mass/volume (e.g. 'to taste'), return null for grams and explain in notes.
Respond ONLY with a JSON object strictly adhering to the provided schema: {{ \"grams\": float_or_null, \"notes\": \"string_explanation\" }}.",
            ingredient.ingredient_name,
            ingredient.quantity,
            ingredient.unit,
            ingredient.preparation_notes
        );

        let request = ChatCompletionRequest {
            model: "qwen/qwen3-32b".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are an expert unit conversion assistant. Output JSON.".to_string(), 
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: conversion_prompt,
                },
            ],
            response_format: Some(ResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: Some(get_gram_conversion_json_schema()),
            }),
            temperature: Some(0.0), 
            max_tokens: Some(150),  
        };

        match provider.call_chat_completion(request).await {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    let mut content_str = choice.message.content.trim().to_string();
                    if content_str.starts_with("```json") && content_str.ends_with("```") {
                        content_str = content_str.trim_start_matches("```json").trim_end_matches("```").trim().to_string();
                    } else if content_str.starts_with("```") && content_str.ends_with("```") {
                        content_str = content_str.trim_start_matches("```").trim_end_matches("```").trim().to_string();
                    }

                    match serde_json::from_str::<GramConversionResponse>(&content_str) {
                        Ok(conv_response) => {
                            progress_updater(format!(
                                " -> Converted: {:?} grams. Notes: {}",
                                conv_response.grams, conv_response.notes
                            ));
                            cleaned_ingredients.push(CleanedIngredient {
                                raw_text: ingredient.raw_text.clone(),
                                ingredient_name: ingredient.ingredient_name.clone(),
                                original_quantity: ingredient.quantity.clone(),
                                original_unit: ingredient.unit.clone(),
                                preparation_notes: ingredient.preparation_notes.clone(),
                                quantity_grams: conv_response.grams,
                                conversion_source: "LLM".to_string(),
                                conversion_notes: Some(conv_response.notes),
                                nutritional_info: None, 
                            });
                        }
                        Err(e) => {
                            progress_updater(format!(
                                " -> Failed to parse LLM conversion response for '{}': {}. Raw: {}",
                                ingredient.ingredient_name, e, content_str
                            ));
                            cleaned_ingredients.push(CleanedIngredient {
                                raw_text: ingredient.raw_text.clone(),
                                ingredient_name: ingredient.ingredient_name.clone(),
                                original_quantity: ingredient.quantity.clone(),
                                original_unit: ingredient.unit.clone(),
                                preparation_notes: ingredient.preparation_notes.clone(),
                                quantity_grams: None,
                                conversion_source: "LLM_Error".to_string(),
                                conversion_notes: Some(format!("Failed to parse LLM response: {}. Raw: {}", e, content_str)),
                                nutritional_info: None, 
                            });
                        }
                    }
                } else {
                    progress_updater(format!(
                        " -> No response choice from LLM for '{}'",
                        ingredient.ingredient_name
                    ));
                     cleaned_ingredients.push(CleanedIngredient {
                        raw_text: ingredient.raw_text.clone(),
                        ingredient_name: ingredient.ingredient_name.clone(),
                        original_quantity: ingredient.quantity.clone(),
                        original_unit: ingredient.unit.clone(),
                        preparation_notes: ingredient.preparation_notes.clone(),
                        quantity_grams: None,
                        conversion_source: "LLM_Error".to_string(),
                        conversion_notes: Some("No response choice from LLM.".to_string()),
                        nutritional_info: None, 
                    });
                }
            }
            Err(e) => {
                progress_updater(format!(
                    " -> API call failed for '{}': {}",
                    ingredient.ingredient_name, e
                ));
                cleaned_ingredients.push(CleanedIngredient {
                    raw_text: ingredient.raw_text.clone(),
                    ingredient_name: ingredient.ingredient_name.clone(),
                    original_quantity: ingredient.quantity.clone(),
                    original_unit: ingredient.unit.clone(),
                    preparation_notes: ingredient.preparation_notes.clone(),
                    quantity_grams: None,
                    conversion_source: "API_Error".to_string(),
                    conversion_notes: Some(format!("API call failed: {}", e)),
                    nutritional_info: None, 
                });
            }
        }
    }

    Ok(CleanedRecipe {
        recipe_title: parsed_recipe.recipe_title.clone(),
        ingredients: cleaned_ingredients,
        instructions: parsed_recipe.instructions.clone(),
    })
}
