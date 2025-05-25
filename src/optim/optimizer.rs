use anyhow::{Result, Context, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::recipe_converter::{CleanedRecipe, CleanedIngredient, convert_ingredients_to_grams, CalculatedNutritionalInfo};
use crate::recipe_parser::{ParsedRecipe, ParsedIngredient}; 
use crate::recipe_aggregator::{calculate_nutritional_profile, RecipeNutritionalProfile, NutritionalSummary};
use crate::nutritional_matcher::NutritionalIndex;
use crate::optim::targets::TargetNutritionalValues;
use crate::optim::nutri_eval::calculate_mse; 
use crate::api_connection::endpoints::{ChatCompletionRequest, ChatMessage, ResponseFormat, JsonSchemaDefinition, JsonSchema, JsonSchemaProperty, Provider};

// --- Structs for LLM Interaction ---

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum LlmOperationType {
    ReplaceIngredient,
    AdjustQuantity,
    AddIngredient,
    RemoveIngredient,
    NoChange,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlmRecipeModification {
    pub operation: LlmOperationType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_ingredient_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacement_description: Option<String>, 
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_ingredient_name: Option<String>, 
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantity_raw: Option<String>, 
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_raw: Option<String>, 
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preparation_notes: Option<String>, 
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlmModificationResponse {
    pub modifications: Vec<LlmRecipeModification>,
    pub overall_reasoning: String,
}

// --- Helper function to apply LLM modifications ---

fn apply_modifications_to_recipe(
    current_recipe: &CleanedRecipe,
    llm_suggestions: &LlmModificationResponse,
    progress_updater: &impl Fn(String),
) -> Result<ParsedRecipe> {
    progress_updater("Applying LLM suggestions to create a candidate recipe...".to_string());
    let mut candidate_ingredients: Vec<ParsedIngredient> = current_recipe.ingredients.iter().map(|ci| {
        let (quantity, unit) = ci.quantity_grams.map_or_else(
            || (ci.original_quantity.clone(), ci.original_unit.clone()),
            |q_g| (format!("{:.1}", q_g), "g".to_string()) 
        );

        ParsedIngredient {
            raw_text: ci.raw_text.clone(),
            ingredient_name: ci.ingredient_name.clone(),
            quantity,
            unit,
            preparation_notes: ci.preparation_notes.clone(),
        }
    }).collect();

    let mut new_ingredients_from_llm: Vec<ParsedIngredient> = Vec::new();

    for modification in &llm_suggestions.modifications {
        progress_updater(format!("  Applying operation: {:?} for {:?}", modification.operation, modification.original_ingredient_name.as_deref().or(modification.replacement_description.as_deref())));
        match modification.operation {
            LlmOperationType::RemoveIngredient => {
                let original_name = modification.original_ingredient_name.as_ref()
                    .ok_or_else(|| anyhow!("'original_ingredient_name' missing for RemoveIngredient operation."))?;
                candidate_ingredients.retain(|ing| &ing.ingredient_name != original_name);
                progress_updater(format!("    Removed ingredient: {}", original_name));
            }
            LlmOperationType::AdjustQuantity => {
                let original_name = modification.original_ingredient_name.as_ref()
                    .ok_or_else(|| anyhow!("'original_ingredient_name' missing for AdjustQuantity operation."))?;
                let new_quantity = modification.quantity_raw.as_ref()
                    .ok_or_else(|| anyhow!("'quantity_raw' missing for AdjustQuantity on '{}'", original_name))?;
                let new_unit = modification.unit_raw.as_ref()
                    .ok_or_else(|| anyhow!("'unit_raw' missing for AdjustQuantity on '{}'", original_name))?;
                
                let mut found = false;
                for ing in candidate_ingredients.iter_mut() {
                    if &ing.ingredient_name == original_name {
                        ing.quantity = new_quantity.clone();
                        ing.unit = new_unit.clone();
                        ing.raw_text = format!("{} {} {}", new_quantity, new_unit, ing.ingredient_name); 
                        if let Some(notes) = &modification.preparation_notes {
                            ing.preparation_notes = notes.clone();
                        }
                        found = true;
                        progress_updater(format!("    Adjusted quantity for {}: to {} {}", original_name, new_quantity, new_unit));
                        break;
                    }
                }
                if !found {
                    progress_updater(format!("    Warning: Ingredient '{}' not found for AdjustQuantity.", original_name));
                }
            }
            LlmOperationType::AddIngredient => {
                let description = modification.replacement_description.as_ref()
                    .ok_or_else(|| anyhow!("'replacement_description' missing for AddIngredient operation."))?;
                let quantity = modification.quantity_raw.as_ref()
                    .ok_or_else(|| anyhow!("'quantity_raw' missing for AddIngredient of '{}'", description))?;
                let unit = modification.unit_raw.as_ref()
                    .ok_or_else(|| anyhow!("'unit_raw' missing for AddIngredient of '{}'", description))?;
                
                let new_parsed_ingredient = ParsedIngredient {
                    raw_text: format!("{} {} {}", quantity, unit, description), 
                    ingredient_name: modification.new_ingredient_name.clone().unwrap_or_else(|| description.clone()), 
                    quantity: quantity.clone(),
                    unit: unit.clone(),
                    preparation_notes: modification.preparation_notes.clone().unwrap_or_default(),
                };
                new_ingredients_from_llm.push(new_parsed_ingredient.clone());
                progress_updater(format!("    Added ingredient: {} {} {}", quantity, unit, description));
            }
            LlmOperationType::ReplaceIngredient => {
                let original_name = modification.original_ingredient_name.as_ref()
                    .ok_or_else(|| anyhow!("'original_ingredient_name' missing for ReplaceIngredient operation."))?;
                let replacement_desc = modification.replacement_description.as_ref()
                    .ok_or_else(|| anyhow!("'replacement_description' missing for ReplaceIngredient of '{}'", original_name))?;
                let quantity = modification.quantity_raw.as_ref()
                    .ok_or_else(|| anyhow!("'quantity_raw' missing for ReplaceIngredient of '{}'", original_name))?;
                let unit = modification.unit_raw.as_ref()
                    .ok_or_else(|| anyhow!("'unit_raw' missing for ReplaceIngredient of '{}'", original_name))?;

                let original_exists = candidate_ingredients.iter().any(|ing| &ing.ingredient_name == original_name);
                if original_exists {
                    candidate_ingredients.retain(|ing| &ing.ingredient_name != original_name);
                    progress_updater(format!("    (Replace) Removed ingredient: {}", original_name));
                } else {
                     progress_updater(format!("    Warning: Original ingredient '{}' for replacement not found.", original_name));
                }
                
                let new_parsed_ingredient = ParsedIngredient {
                    raw_text: format!("{} {} {}", quantity, unit, replacement_desc),
                    ingredient_name: modification.new_ingredient_name.clone().unwrap_or_else(|| replacement_desc.clone()),
                    quantity: quantity.clone(),
                    unit: unit.clone(),
                    preparation_notes: modification.preparation_notes.clone().unwrap_or_default(),
                };
                new_ingredients_from_llm.push(new_parsed_ingredient.clone());
                progress_updater(format!("    (Replace) Added ingredient: {} {} {}", quantity, unit, replacement_desc));
            }
            LlmOperationType::NoChange => {
                progress_updater("    NoChange operation encountered within apply_modifications. This is unexpected here.".to_string());
            }
        }
    }
    
    candidate_ingredients.extend(new_ingredients_from_llm);

    Ok(ParsedRecipe {
        recipe_title: current_recipe.recipe_title.clone(), 
        ingredients: candidate_ingredients,
        instructions: current_recipe.instructions.clone(), 
    })
}

// --- Main Optimization Function ---

pub async fn optimize_recipe(
    initial_cleaned_recipe: &CleanedRecipe,
    initial_nutritional_profile: &RecipeNutritionalProfile,
    target_nutrition_per_100g: &TargetNutritionalValues,
    max_iterations: u32,
    nutritional_index: &NutritionalIndex,
    api_key_env_var: &str,
    progress_updater: impl Fn(String) + Send + Sync + Clone + 'static,
) -> Result<CleanedRecipe> {
    progress_updater(format!("Starting recipe optimization. Max iterations: {}", max_iterations));
    progress_updater(format!("Initial recipe title: {}", initial_cleaned_recipe.recipe_title));
    progress_updater(format!("Target nutrition (per 100g): {:?}", target_nutrition_per_100g));

    let mut current_best_recipe = initial_cleaned_recipe.clone();
    let mut current_best_profile = initial_nutritional_profile.clone();
    let mut current_best_mse = calculate_mse(&current_best_profile.per_100g, target_nutrition_per_100g);
    progress_updater(format!("Initial MSE: {:.4}", current_best_mse));

    for i in 0..max_iterations {
        progress_updater(format!("\n--- Optimization Iteration {}/{} ---", i + 1, max_iterations));

        // 1. Construct Prompt for LLM
        let system_prompt = format!(
            "/no_thinking
You are a recipe optimization assistant. Your goal is to modify the given recipe to meet specific nutritional targets while maintaining or improving palatability and culinary coherence.
Output your suggested modifications as a JSON object.
The JSON object must be the only content in your response. Do not include any explanatory text, comments, or markdown formatting (like ```json) before or after the JSON object.
Your response must start with {{{{ and end with }}}}.

The JSON object MUST adhere to the 'recipe_modification_suggestions' schema provided to you.
The 'modifications' array MUST contain **EXACTLY ONE** modification object.
Example of the required structure:
{{{{
  \"modifications\": [
    {{ \"operation\": \"replace_ingredient\", \"original_ingredient_name\": \"example original\", \"replacement_description\": \"example replacement\", \"quantity_raw\": \"100\", \"unit_raw\": \"g\", \"reasoning\": \"This single change is most impactful.\" }}
  ],
  \"overall_reasoning\": \"This is the overall explanation for why this single change helps meet the target.\"
}}}}
Do NOT nest this structure inside any other keys.
The 'overall_reasoning' field MUST be a string at the top level.

**CRITICAL RULE: You MUST suggest EXACTLY ONE modification in the 'modifications' array.**
This single modification should be the one you believe will have the most positive impact on reducing the MSE towards the target nutritional profile, while being culinarily sensible.

Current MSE (Mean Squared Error) from target: {:.4} (lower is better). Aim to reduce this with your single suggested change.
**Strategy Guidance for your SINGLE modification:**
- **Highest Impact:** Choose the single change (replace, adjust, add, or remove an ingredient) that you predict will best improve the nutritional profile towards the targets.
- **Culinary Sense:** The change MUST make sense for the recipe type.
- **Targeted Modification:** If a specific macronutrient is far from target, your single change should ideally address that.
- **No Change (as the single operation):** If you believe the recipe is already optimal or any single change would be detrimental, you can use the 'no_change' operation as your single modification.

Consider the following operations for your **SINGLE** modification:
- 'replace_ingredient': Swap an existing ingredient with another.
- 'adjust_quantity': Change the amount of an existing ingredient.
- 'add_ingredient': Introduce a new ingredient.
- 'remove_ingredient': Delete an ingredient.
- 'no_change': Use this if no single beneficial change can be identified.

When suggesting quantities and units for your single modification:
- For 'quantity_raw', provide a string that can be parsed as a number or a common textual quantity.
- For 'unit_raw', provide a common unit.

The 'Current Recipe Ingredients' list below shows ingredients with their quantities primarily in grams (g).
Focus on macronutrient targets (protein, carbohydrates, fat). Kcal is derived.
The 'original_ingredient_name' for any modification MUST EXACTLY MATCH one of the ingredient names from the 'Current Recipe Ingredients' list.
",
        current_best_mse 
        );

        let current_ingredients_text = current_best_recipe.ingredients.iter()
            .map(|ing| {
                let quantity_display = ing.quantity_grams.map_or_else( 
                    || ing.raw_text.clone(), 
                    |q_g| format!("{:.1} g", q_g) 
                );
                format!("- {} (Current Quantity: {}, Original Text: '{}')", 
                    ing.ingredient_name, 
                    quantity_display,
                    ing.raw_text 
                )
            })
            .collect::<Vec<String>>()
            .join("\n");

        let opt_f32_to_str = |val: Option<f32>| val.map_or_else(|| "N/A".to_string(), |v| format!("{:.1}", v));

        let user_prompt_content = format!(
"Current Recipe Title: {}

Current Recipe Ingredients:
{}

Current Nutritional Profile (per 100g):
- Kcal: {}
- Protein: {} g
- Carbohydrates: {} g
- Fat: {} g
- Sugars: {} g (for reference)
- Saturated Fat: {} g (for reference)
- Salt: {} g (for reference)

Target Nutritional Profile (per 100g):
- Kcal: {} (estimate, nutriments are more important)
- Protein: {} g
- Carbohydrates: {} g
- Fat: {} g

Please suggest **EXACTLY ONE** modification to the recipe to bring its nutritional profile closer to the target values, aiming to reduce the MSE, following the strategy guidance for a single change.
Return your suggestion in the specified JSON format (modifications array must have only one item).
",
            current_best_recipe.recipe_title,
            current_ingredients_text,
            opt_f32_to_str(current_best_profile.per_100g.kcal),
            opt_f32_to_str(current_best_profile.per_100g.protein_g),
            opt_f32_to_str(current_best_profile.per_100g.carbohydrate_g),
            opt_f32_to_str(current_best_profile.per_100g.fat_g),
            opt_f32_to_str(current_best_profile.per_100g.sugars_g),
            opt_f32_to_str(current_best_profile.per_100g.fa_saturated_g),
            opt_f32_to_str(current_best_profile.per_100g.salt_g),
            opt_f32_to_str(target_nutrition_per_100g.kcal),
            opt_f32_to_str(target_nutrition_per_100g.protein_g),
            opt_f32_to_str(target_nutrition_per_100g.carbohydrate_g),
            opt_f32_to_str(target_nutrition_per_100g.fat_g),
        );
        
        progress_updater(format!("System Prompt (Iteration {}):\n{}", i + 1, system_prompt));
        progress_updater(format!("User Prompt (Iteration {}):\n{}", i + 1, user_prompt_content));

        // 2. Call LLM
        let provider = Provider::openrouter(api_key_env_var);
        let llm_schema = get_llm_modification_schema_single_item(); // Use a schema that expects a single item

        let request = ChatCompletionRequest {
            model: "qwen/qwen3-32b".to_string(), 
            messages: vec![
                ChatMessage { role: "system".to_string(), content: system_prompt },
                ChatMessage { role: "user".to_string(), content: user_prompt_content },
            ],
            response_format: Some(ResponseFormat {
                format_type: "json_object".to_string(), 
                json_schema: Some(llm_schema),
            }),
            temperature: Some(0.1), // Lowered temperature further
            max_tokens: Some(1024), // Reduced max_tokens
        };

        progress_updater(format!("Sending request to LLM (Iteration {})...", i + 1));
        
        let llm_response_str = match provider.call_chat_completion(request).await {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    progress_updater(format!("LLM Response (Iteration {}):\n{}", i + 1, choice.message.content));
                    choice.message.content.clone()
                } else {
                    return Err(anyhow!("LLM returned no choices in response."));
                }
            }
            Err(e) => {
                progress_updater(format!("LLM call failed (Iteration {}): {}", i + 1, e));
                eprintln!("LLM call failed: {}. Using mock 'no_change' response.", e);
                 r#"{
                    "modifications": [ { "operation": "no_change", "reasoning": "LLM call failed, attempting graceful exit." } ],
                    "overall_reasoning": "LLM call failed during optimization."
                }"#.to_string()
            }
        };
        
        let llm_suggestion: LlmModificationResponse = match serde_json::from_str::<LlmModificationResponse>(&llm_response_str) { // Added Turbofish
            Ok(mut suggestion) => {
                // Ensure only one modification is processed, even if LLM violates prompt
                if suggestion.modifications.len() > 1 {
                    progress_updater(format!("Warning: LLM returned {} modifications, but prompt asked for 1. Taking only the first.", suggestion.modifications.len()));
                    suggestion.modifications.truncate(1);
                }
                if suggestion.modifications.is_empty() && !llm_response_str.contains("no_change") { // If it's empty but wasn't a deliberate no_change
                     progress_updater(format!("LLM returned empty modifications array. Interpreting as 'no_change'. Content: {}", llm_response_str));
                     suggestion.modifications.push(LlmRecipeModification {
                        operation: LlmOperationType::NoChange,
                        reasoning: Some("LLM returned empty modifications, interpreted as no change.".to_string()),
                        ..Default::default() // Fill with None/Default
                     });
                }
                suggestion
            }
            Err(e) => {
                progress_updater(format!("Failed to parse LLM suggestion (Iteration {}): {}. Content: '{}'", i + 1, e, llm_response_str));
                // Fallback to no_change if parsing fails completely
                LlmModificationResponse {
                    modifications: vec![LlmRecipeModification {
                        operation: LlmOperationType::NoChange,
                        reasoning: Some(format!("Failed to parse LLM JSON output: {}. Content: '{}'", e, llm_response_str)),
                        ..Default::default()
                    }],
                    overall_reasoning: format!("Failed to parse LLM JSON output: {}. Content: '{}'", e, llm_response_str),
                }
            }
        };

        if llm_suggestion.modifications.is_empty() || 
           (llm_suggestion.modifications.len() == 1 && matches!(llm_suggestion.modifications[0].operation, LlmOperationType::NoChange)) {
            progress_updater(format!("LLM suggested no changes or failed to provide valid changes. Reason: {}. Ending optimization.", 
                llm_suggestion.modifications.first().and_then(|m| m.reasoning.as_ref()).map_or(
                    llm_suggestion.overall_reasoning.as_str(),
                    |s| s.as_str()
                )
            ));
            break;
        }
        
        let candidate_parsed_recipe = match apply_modifications_to_recipe(&current_best_recipe, &llm_suggestion, &progress_updater) {
            Ok(recipe) => recipe,
            Err(e) => {
                progress_updater(format!("Error applying LLM modifications: {}. Skipping this iteration.", e));
                continue; 
            }
        };
        
        progress_updater("Converting candidate recipe ingredients to grams...".to_string());
        let mut candidate_cleaned_recipe = match convert_ingredients_to_grams(&candidate_parsed_recipe, api_key_env_var, progress_updater.clone()).await {
            Ok(recipe) => recipe,
            Err(e) => {
                progress_updater(format!("Error converting candidate ingredients to grams: {}. Skipping this iteration.", e));
                continue;
            }
        };

        progress_updater("Enriching candidate recipe with nutritional information...".to_string());
        for ingredient in candidate_cleaned_recipe.ingredients.iter_mut() {
            if ingredient.quantity_grams.is_some() { 
                match nutritional_index.find_and_calculate_nutrition(ingredient, api_key_env_var, &progress_updater).await {
                    Ok(Some(calculated_info)) => { 
                        ingredient.nutritional_info = Some(calculated_info); 
                        progress_updater(format!("  -> Successfully enriched '{}'", ingredient.ingredient_name));
                    }
                    Ok(None) => {
                        progress_updater(format!("  -> Could not find nutritional info for '{}'", ingredient.ingredient_name));
                    }
                    Err(e) => {
                        progress_updater(format!("  -> Error enriching '{}': {}", ingredient.ingredient_name, e));
                    }
                }
            }
        }

        let candidate_profile = calculate_nutritional_profile(&candidate_cleaned_recipe);
        progress_updater(format!("Candidate recipe nutritional profile (per 100g): Kcal: {}, P: {}, C: {}, F: {}",
            opt_f32_to_str(candidate_profile.per_100g.kcal),
            opt_f32_to_str(candidate_profile.per_100g.protein_g),
            opt_f32_to_str(candidate_profile.per_100g.carbohydrate_g),
            opt_f32_to_str(candidate_profile.per_100g.fat_g)
        ));

        let candidate_mse = calculate_mse(&candidate_profile.per_100g, target_nutrition_per_100g);
        progress_updater(format!("Candidate MSE: {:.4}", candidate_mse));

        if candidate_mse < current_best_mse {
            progress_updater(format!("Found improved recipe. New MSE: {:.4} (was {:.4})", candidate_mse, current_best_mse));
            current_best_recipe = candidate_cleaned_recipe;
            current_best_profile = candidate_profile;
            current_best_mse = candidate_mse;
        } else {
            progress_updater(format!("Candidate recipe did not improve MSE (Candidate: {:.4}, Best: {:.4}). Retaining previous best.", candidate_mse, current_best_mse));
        }
    }

    progress_updater(format!("\nOptimization finished. Best recipe found: {} with MSE: {:.4}", current_best_recipe.recipe_title, current_best_mse));
    
    Ok(current_best_recipe)
}

// Schema for a single modification item in the array
fn get_llm_modification_schema_single_item() -> JsonSchemaDefinition {
    let operation_type_enum = vec![
        "replace_ingredient".to_string(),
        "adjust_quantity".to_string(),
        "add_ingredient".to_string(),
        "remove_ingredient".to_string(),
        "no_change".to_string(),
    ];

    let mut modification_properties = HashMap::new();
    modification_properties.insert(
        "operation".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("The type of modification to perform.".to_string()),
            r#enum: Some(operation_type_enum),
            items: None,
        },
    );
     modification_properties.insert(
        "original_ingredient_name".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Name of the ingredient to modify/remove. Required for all operations except 'add_ingredient' if it's a truly new item not replacing anything. For 'no_change', can be omitted or refer to the whole recipe concept.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "replacement_description".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Descriptive name for 'replace' or 'add' operations.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "new_ingredient_name".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Specific name of a new ingredient if known (for add/replace).".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "quantity_raw".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("New quantity for 'replace', 'adjust', 'add' operations.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "unit_raw".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("New unit for 'replace', 'adjust', 'add' operations.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "preparation_notes".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Optional preparation notes.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "reasoning".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Brief reasoning for this specific modification.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    
    let modification_schema = JsonSchema {
        schema_type: "object".to_string(),
        properties: Some(modification_properties),
        required: Some(vec!["operation".to_string()]), 
        additional_properties: Some(true), 
    };

    let mut response_properties = HashMap::new();
    response_properties.insert(
        "modifications".to_string(),
        JsonSchemaProperty {
            property_type: "array".to_string(),
            description: Some("A list containing EXACTLY ONE suggested modification to the recipe.".to_string()),
            items: Some(Box::new(modification_schema.clone())),
            r#enum: None,
        },
    );
    response_properties.insert(
        "overall_reasoning".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Overall reasoning for the single suggested modification.".to_string()),
            r#enum: None,
            items: None,
        },
    );

    JsonSchemaDefinition {
        name: "recipe_modification_suggestions_single_item".to_string(), 
        strict: Some(true), 
        schema: JsonSchema {
            schema_type: "object".to_string(),
            properties: Some(response_properties),
            required: Some(vec!["modifications".to_string(), "overall_reasoning".to_string()]),
            additional_properties: Some(false), 
        },
    }
}

#[allow(dead_code)]
fn get_llm_modification_schema() -> JsonSchemaDefinition {
    let operation_type_enum = vec![
        "replace_ingredient".to_string(),
        "adjust_quantity".to_string(),
        "add_ingredient".to_string(),
        "remove_ingredient".to_string(),
        "no_change".to_string(),
    ];

    let mut modification_properties = HashMap::new();
    modification_properties.insert(
        "operation".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("The type of modification to perform.".to_string()),
            r#enum: Some(operation_type_enum),
            items: None,
        },
    );
    modification_properties.insert(
        "original_ingredient_name".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Name of the ingredient to modify/remove (for replace, adjust, remove operations). Must exactly match an ingredient name from the provided list.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "replacement_description".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Descriptive name of the ingredient to use as replacement or to add (for replace, add operations). E.g., 'low-fat Greek yogurt', 'whole wheat flour'. The system will try to find a match in its database.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "new_ingredient_name".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Specific name of a new ingredient if known, otherwise use replacement_description (for add operation, or if LLM is very sure about a replacement's specific name).".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "quantity_raw".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("New quantity for the ingredient (for replace, adjust, add operations). E.g., '1', '0.5', '200'. Should be a numerical value or common textual quantity.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "unit_raw".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("New unit for the ingredient (for replace, adjust, add operations). E.g., 'cup', 'g', 'ml', 'tbsp', 'piece'. Should be a common unit abbreviation or full name.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "preparation_notes".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Optional preparation notes for the new/modified ingredient (e.g., 'sifted', 'finely chopped').".to_string()),
            r#enum: None,
            items: None,
        },
    );
    modification_properties.insert(
        "reasoning".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Brief reasoning for this specific modification.".to_string()),
            r#enum: None,
            items: None,
        },
    );
    
    let modification_schema = JsonSchema {
        schema_type: "object".to_string(),
        properties: Some(modification_properties),
        required: Some(vec!["operation".to_string()]), 
        additional_properties: Some(true), 
    };

    let mut response_properties = HashMap::new();
    response_properties.insert(
        "modifications".to_string(),
        JsonSchemaProperty {
            property_type: "array".to_string(),
            description: Some("A list of suggested modifications to the recipe. Each modification must be an object.".to_string()),
            items: Some(Box::new(modification_schema.clone())), 
            r#enum: None,
        },
    );
    response_properties.insert(
        "overall_reasoning".to_string(),
        JsonSchemaProperty {
            property_type: "string".to_string(),
            description: Some("Overall reasoning for the suggested set of modifications, explaining how they aim to meet the nutritional targets.".to_string()),
            r#enum: None,
            items: None,
        },
    );

    JsonSchemaDefinition {
        name: "recipe_modification_suggestions".to_string(),
        strict: Some(true),
        schema: JsonSchema {
            schema_type: "object".to_string(),
            properties: Some(response_properties),
            required: Some(vec!["modifications".to_string(), "overall_reasoning".to_string()]),
            additional_properties: Some(false),
        },
    }
}

impl Default for LlmRecipeModification {
    fn default() -> Self {
        LlmRecipeModification {
            operation: LlmOperationType::NoChange, 
            original_ingredient_name: None,
            replacement_description: None,
            new_ingredient_name: None,
            quantity_raw: None,
            unit_raw: None,
            preparation_notes: None,
            reasoning: None,
        }
    }
}
