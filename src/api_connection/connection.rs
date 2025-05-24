use dotenv::dotenv;
use reqwest::Client;
use serde_json::json;
use std::env;
use std::error::Error;
use std::fmt;

use super::endpoints::{
    ChatCompletionRequest, ChatCompletionResponse, OpenRouterAvailableModel, Provider,
    OPENROUTER_MODELS,
};

#[derive(Debug)]
pub enum ApiConnectionError {
    MissingApiKey(String),
    NetworkError(reqwest::Error),
    SerializationError(serde_json::Error),
    ApiError {
        status: reqwest::StatusCode,
        error_body: String,
    },
    UnsupportedProvider(String),
}

impl fmt::Display for ApiConnectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiConnectionError::MissingApiKey(key_name) => {
                write!(f, "API key not found in environment: {}", key_name)
            }
            ApiConnectionError::NetworkError(err) => write!(f, "Network error: {}", err),
            ApiConnectionError::SerializationError(err) => {
                write!(f, "Serialization error: {}", err)
            }
            ApiConnectionError::ApiError { status, error_body } => {
                write!(f, "API error {}: {}", status, error_body)
            }
            ApiConnectionError::UnsupportedProvider(provider_name) => {
                write!(f, "Unsupported provider: {}", provider_name)
            }
        }
    }
}

impl Error for ApiConnectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ApiConnectionError::NetworkError(err) => Some(err),
            ApiConnectionError::SerializationError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for ApiConnectionError {
    fn from(err: reqwest::Error) -> Self {
        ApiConnectionError::NetworkError(err)
    }
}

impl From<serde_json::Error> for ApiConnectionError {
    fn from(err: serde_json::Error) -> Self {
        ApiConnectionError::SerializationError(err)
    }
}

impl Provider {
    pub fn openrouter(api_key_env_var_name: &str) -> Self {
        dotenv().ok();
        Self::OpenRouter {
            api_key: api_key_env_var_name.to_string(),
            available_models: OPENROUTER_MODELS.to_vec(),
        }
    }

    pub fn get_available_models(&self) -> Vec<OpenRouterAvailableModel> {
        match self {
            Provider::OpenRouter {
                available_models, ..
            } => available_models.clone(),
        }
    }

    pub async fn call_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ApiConnectionError> {
        match self {
            Provider::OpenRouter {
                api_key: api_key_env_var_name,
                ..
            } => {
                dotenv().ok();
                let actual_api_key = env::var(api_key_env_var_name)
                    .map_err(|_| ApiConnectionError::MissingApiKey(api_key_env_var_name.clone()))?;

                let client = Client::new();
                let url = "https://openrouter.ai/api/v1/chat/completions";

                let mut request_payload = serde_json::to_value(&request)
                    .map_err(ApiConnectionError::SerializationError)?;

                if let Some(obj) = request_payload.as_object_mut() {
                    obj.insert(
                        "provider".to_string(),
                        json!({ "only": ["Cerebras"] }),
                    );
                } else {
                    return Err(ApiConnectionError::SerializationError(
                        serde_json::from_str::<serde_json::Value>(
                            "Failed to create JSON object from request",
                        )
                        .unwrap_err(),
                    ));
                }
                
                let site_url = env::var("SITE_URL").unwrap_or_else(|_| "http://localhost:3000".to_string());
                let app_name = env::var("APP_NAME").unwrap_or_else(|_| "RecipeOptim".to_string());

                let response = client
                    .post(url)
                    .bearer_auth(actual_api_key)
                    .header("Content-Type", "application/json")
                    .header("HTTP-Referer", site_url) 
                    .header("X-Title", app_name)
                    .json(&request_payload)
                    .send()
                    .await?;

                if response.status().is_success() {
                    let chat_response = response.json::<ChatCompletionResponse>().await?;
                    Ok(chat_response)
                } else {
                    let status = response.status();
                    let error_body = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Failed to read error body".to_string());
                    Err(ApiConnectionError::ApiError { status, error_body })
                }
            }
        }
    }
}
