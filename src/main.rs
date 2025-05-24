use anyhow::{Result, Context};
use recipe_optim::cli::parse_args;
use recipe_optim::recipe_parser::parse_recipe_text;
use recipe_optim::recipe_converter::{convert_ingredients_to_grams, CleanedRecipe}; // Added CleanedRecipe
use recipe_optim::nutritional_matcher::NutritionalIndex; // Added import
use tokio::fs;
use std::path::Path; // For Path type

// Define the environment variable name for the API key
const API_KEY_ENV_VAR: &str = "OPENROUTER_API_KEY";
const CIQUAL_CSV_PATH: &str = "ciqual.csv"; // Define path to ciqual.csv

async fn enrich_with_nutritional_info(
    cleaned_recipe: &mut CleanedRecipe, // Take mutable reference to update in place
    nutritional_index: &NutritionalIndex,
    api_key_env_var: &str,
    progress_updater: impl Fn(String) + Send + Sync + 'static,
) -> Result<()> {
    println!("\nEnriching recipe with nutritional information...");
    let ingredients_count = cleaned_recipe.ingredients.len(); // Get length before mutable borrow
    for (idx, ingredient) in cleaned_recipe.ingredients.iter_mut().enumerate() {
        progress_updater(format!(
            "Processing ingredient {}/{} for nutrition: {}",
            idx + 1,
            ingredients_count, // Use the stored length
            ingredient.ingredient_name
        ));
        
        match nutritional_index.find_and_calculate_nutrition(ingredient, api_key_env_var, &progress_updater).await {
            Ok(Some(nutritional_info)) => {
                progress_updater(format!(
                    "   -> Successfully calculated nutrition for '{}' from Ciqual item: '{}'",
                    ingredient.ingredient_name, nutritional_info.source_ciqual_name
                ));
                ingredient.nutritional_info = Some(nutritional_info);
            }
            Ok(None) => {
                progress_updater(format!(
                    "   -> Could not find or calculate nutritional information for '{}'",
                    ingredient.ingredient_name
                ));
            }
            Err(e) => {
                 progress_updater(format!(
                    "   -> Error finding nutrition for '{}': {}",
                    ingredient.ingredient_name, e
                ));
                // Optionally, decide if this error should halt everything or just skip the ingredient
            }
        }
    }
    println!("Nutritional enrichment complete.");
    Ok(())
}


#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok(); // Load .env file for API keys

    let cli_args = parse_args();
    println!("Attempting to read recipe file: {}", cli_args.recipe_file);

    // Initialize NutritionalIndex once
    // This can take time due to loading, embedding, and ANN building.
    println!("Initializing Nutritional Index (this may take a moment)...");
    let nutritional_index = NutritionalIndex::new(Path::new(CIQUAL_CSV_PATH), API_KEY_ENV_VAR)
        .with_context(|| format!("Failed to initialize Nutritional Index with Ciqual data from '{}'", CIQUAL_CSV_PATH))?;
    println!("Nutritional Index initialized.");


    let recipe_content = fs::read_to_string(&cli_args.recipe_file)
        .await
        .with_context(|| format!("Failed to read recipe file '{}'", cli_args.recipe_file))?;

    println!("\nRecipe content read successfully. Sending to parser...");

    match parse_recipe_text(&recipe_content, API_KEY_ENV_VAR).await {
        Ok(parsed_recipe) => {
            println!("\nSuccessfully parsed recipe. Now converting ingredients to grams...");
            
            let progress_callback = |message: String| {
                println!("{}", message);
            };

            match convert_ingredients_to_grams(&parsed_recipe, API_KEY_ENV_VAR, progress_callback).await {
                Ok(mut cleaned_recipe) => { // Made cleaned_recipe mutable
                    println!("\nSuccessfully converted recipe ingredients to grams.");
                    
                    // Enrich with nutritional information
                    if let Err(e) = enrich_with_nutritional_info(&mut cleaned_recipe, &nutritional_index, API_KEY_ENV_VAR, progress_callback).await {
                        eprintln!("\nError enriching recipe with nutritional info: {}", e);
                        // Decide if to proceed or return error
                    }

                    println!("\nFinal Enriched Recipe:");
                    println!("{:#?}", cleaned_recipe);
                }
                Err(e) => {
                    eprintln!("\nError converting recipe ingredients: {}", e);
                    return Err(e.into()); // Convert anyhow::Error from conversion to main's Result
                }
            }
        }
        Err(e) => {
            eprintln!("\nError parsing recipe: {}", e);
            // The error from parse_recipe_text is ApiConnectionError, needs to be mapped or handled
            // For now, let's wrap it with anyhow
            return Err(anyhow::anyhow!("Recipe parsing failed: {}", e));
        }
    }

    Ok(())
}
