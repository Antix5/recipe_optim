use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenRouterAvailableModel {
    pub model_name: &'static str,
    pub model_source: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub enum Provider {
    OpenRouter {
        api_key: String,
        available_models: Vec<OpenRouterAvailableModel>,
    },
}

pub const OPENROUTER_MODELS: &[OpenRouterAvailableModel] = &[
    OpenRouterAvailableModel {
        model_name: "qwen/qwen3-32b",
        model_source: "cerebras",
    },
];

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JsonSchemaProperty {
    #[serde(rename = "type")]
    pub property_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#enum: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JsonSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    pub properties: HashMap<String, JsonSchemaProperty>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub required: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "additionalProperties")]
    pub additional_properties: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JsonSchemaDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    pub schema: JsonSchema,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchemaDefinition>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatCompletionResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatCompletionChoice {
    pub message: ChatCompletionResponseMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    pub index: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: Option<u32>,
    pub total_tokens: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatCompletionUsage>,
}
