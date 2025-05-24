use serde::{Deserialize, Serialize};
use std::collections::HashMap; 
use crate::api_connection::endpoints::{
    ChatCompletionRequest, ChatMessage, JsonSchema, JsonSchemaDefinition, JsonSchemaProperty,
    Provider, // ResponseFormat no longer needed here for parse_recipe_text
};
use crate::api_connection::connection::ApiConnectionError; 
use anyhow::Result;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParsedIngredient {
    pub raw_text: String,
    pub ingredient_name: String,
    pub quantity: String,
    pub unit: String,
    pub preparation_notes: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParsedRecipe {
    #[serde(alias = "title")] 
    pub recipe_title: String,
    pub ingredients: Vec<ParsedIngredient>,
    pub instructions: Vec<String>,
}

// This function might become unused by parse_recipe_text if we fully remove schema enforcement.
// However, it's kept for potential future use or if other parts of the system use it.
// If it's confirmed to be unused, it can be removed later.
#[allow(dead_code)]
fn get_recipe_json_schema() -> JsonSchemaDefinition {
    let ingredient_item_schema = JsonSchema {
        schema_type: "object".to_string(),
        properties: None,
        required: None,
        additional_properties: None, 
    };

    let instruction_item_schema = JsonSchema {
        schema_type: "string".to_string(),
        properties: None, 
        required: None,       
        additional_properties: None,
    };

    let mut recipe_properties_map = HashMap::new();
    recipe_properties_map.insert(
        "recipe_title".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("The title of the recipe.".to_string()),
            r#enum: None,
            items: None, 
        },
    );
    recipe_properties_map.insert(
        "ingredients".to_string(),
        JsonSchemaProperty {
            property_type: "array".to_string(),
            description: Some("A list of ingredients. Each item in the array must be an object with the following string properties: 'raw_text', 'ingredient_name', 'quantity', 'unit', and 'preparation_notes'.".to_string()),
            items: Some(Box::new(ingredient_item_schema)), 
            r#enum: None,
        },
    );
     recipe_properties_map.insert(
        "instructions".to_string(),
        JsonSchemaProperty {
            property_type: "array".to_string(),
            description: Some("A list of cooking instructions.".to_string()),
            items: Some(Box::new(instruction_item_schema)),
            r#enum: None,
        },
    );

    JsonSchemaDefinition {
        name: "parsed_recipe_schema".to_string(),
        strict: Some(true),
        schema: JsonSchema {
            schema_type: "object".to_string(),
            properties: Some(recipe_properties_map),
            required: Some(vec![
                "recipe_title".to_string(),
                "ingredients".to_string(),
                "instructions".to_string(),
            ]),
            additional_properties: Some(false),
        },
    }
}

pub async fn parse_recipe_text(recipe_text: &str, api_key_env_var: &str) -> Result<ParsedRecipe, ApiConnectionError> {
    let system_prompt = format!(
        "/no_thinking
You are a recipe parsing assistant. Your task is to parse the given recipe text and extract its title, ingredients, and instructions.
Return the output as a JSON object. The JSON object must be the only content in your response. Do not include any explanatory text, comments, or markdown formatting (like ```json) before or after the JSON object.
The JSON object must have the following top-level properties:
- \"recipe_title\": A string representing the title of the recipe.
- \"ingredients\": An array of objects. Each object in this array represents a single ingredient.
- \"instructions\": An array of strings, where each string is a distinct cooking instruction.

Each object in the \"ingredients\" array must have the following string properties:
- \"raw_text\": The full, original text of the ingredient line.
- \"ingredient_name\": The primary name of the food item (e.g., 'all-purpose flour', 'egg', 'garlic clove').
- \"quantity\": The amount specified (e.g., '2', '1/2', 'a pinch', '1-2').
- \"unit\": The unit of measurement (e.g., 'cups', 'g', 'ml', 'large', 'clove', 'piece', or an empty string if unitless or descriptive like 'to taste').
- \"preparation_notes\": Any additional notes on preparation or state (e.g., 'sifted', 'finely chopped', 'at room temperature', 'optional', or an empty string if none).

Ensure all specified fields are present in your JSON output. If a piece of information for an optional field (like 'preparation_notes' or 'unit' if not applicable) is not present in the recipe text, use an empty string for that field.
Your response must start with {{ and end with }}.
"
    );

    let provider = Provider::openrouter(api_key_env_var);

    let request = ChatCompletionRequest {
        model: "qwen/qwen3-32b".to_string(), 
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt,
            },
            ChatMessage {
                role: "user".to_string(),
                content: recipe_text.to_string(),
            },
        ],
        response_format: None, // <<<< KEY CHANGE: No json_schema enforcement by the API
        temperature: Some(0.05), 
        max_tokens: Some(2048), 
    };

    let response = provider.call_chat_completion(request).await?;

    if let Some(choice) = response.choices.first() {
        let mut content_str = choice.message.content.trim().to_string(); 
        println!("[DEBUG] Raw API Response Content:\n---\n{}\n---", content_str);

        // Attempt to strip markdown code fences if present
        if content_str.starts_with("```json") && content_str.ends_with("```") {
            content_str = content_str.trim_start_matches("```json").trim_end_matches("```").trim().to_string();
            println!("[DEBUG] Content after stripping '```json...```':\n---\n{}\n---", content_str);
        } else if content_str.starts_with("```") && content_str.ends_with("```") {
            content_str = content_str.trim_start_matches("```").trim_end_matches("```").trim().to_string();
            println!("[DEBUG] Content after stripping '```...```':\n---\n{}\n---", content_str);
        }
        
        if content_str.is_empty() {
            eprintln!("[DEBUG] API response content is empty after stripping markdown.");
            return Err(ApiConnectionError::ApiError {
                status: reqwest::StatusCode::NO_CONTENT, 
                error_body: "API returned empty content after stripping markdown.".to_string(),
            });
        }
        
        // The LLM might still not return perfect JSON, so this parsing can still fail.
        serde_json::from_str(&content_str) 
            .map_err(|e| {
                eprintln!("[DEBUG] Failed to deserialize content. Error: {}. Content was:\n{}", e, content_str);
                ApiConnectionError::SerializationError(e)
            })
    } else {
        eprintln!("[DEBUG] No choices received from API response.");
        Err(ApiConnectionError::ApiError { 
            status: reqwest::StatusCode::INTERNAL_SERVER_ERROR, 
            error_body: "No response choices received from API".to_string(),
        })
    }
}
