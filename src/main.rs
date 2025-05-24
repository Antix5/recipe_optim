use anyhow::{Result, Context, bail, anyhow}; 
use recipe_optim::cli::parse_args;
use recipe_optim::recipe_parser::{parse_recipe_text, ParsedRecipe};
use recipe_optim::recipe_converter::{convert_ingredients_to_grams, CleanedRecipe};
use recipe_optim::nutritional_matcher::NutritionalIndex;
use recipe_optim::recipe_aggregator::{calculate_nutritional_profile, EnrichedRecipeOutput, RecipeNutritionalProfile};
use recipe_optim::optim::targets::{calculate_target_nutrition, TargetNutritionalValues}; 
use recipe_optim::optim::optimizer::optimize_recipe; 
use tokio::fs;
use std::path::{Path, PathBuf};

// Define the environment variable name for the API key
const API_KEY_ENV_VAR: &str = "OPENROUTER_API_KEY";
const CIQUAL_CSV_PATH: &str = "ciqual.csv"; // Define path to ciqual.csv

async fn enrich_with_nutritional_info(
    cleaned_recipe: &mut CleanedRecipe, 
    nutritional_index: &NutritionalIndex,
    api_key_env_var: &str,
    progress_updater: impl Fn(String) + Send + Sync + 'static,
) -> Result<()> {
    println!("\nEnriching recipe with nutritional information...");
    let ingredients_count = cleaned_recipe.ingredients.len();
    for (idx, ingredient) in cleaned_recipe.ingredients.iter_mut().enumerate() {
        progress_updater(format!(
            "Processing ingredient {}/{} for nutrition: {}",
            idx + 1,
            ingredients_count,
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
    println!("Input recipe file: {}", cli_args.recipe_file);

    let input_path = PathBuf::from(&cli_args.recipe_file);
    let file_stem = input_path.file_stem().unwrap_or_default().to_string_lossy();
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    
    let enriched_file_name = format!("{}_enriched.json", file_stem);
    let enriched_file_path = parent_dir.join(&enriched_file_name);
    let optimized_file_name = format!("{}_optimized.json", file_stem); 
    let optimized_file_path = parent_dir.join(&optimized_file_name);

    let mut initial_cleaned_recipe_opt: Option<CleanedRecipe> = None;
    let mut initial_nutritional_profile_opt: Option<RecipeNutritionalProfile> = None;
    
    // Attempt to load existing enriched file first
    if enriched_file_path.exists() { 
        println!("Attempting to load existing enriched file: {:?}", enriched_file_path);
        let enriched_content = fs::read_to_string(&enriched_file_path).await
            .with_context(|| format!("Failed to read existing enriched file {:?}", enriched_file_path))?;
        
        match serde_json::from_str::<EnrichedRecipeOutput>(&enriched_content) {
            Ok(loaded_data) => {
                println!("Successfully loaded and parsed existing enriched data.");
                initial_cleaned_recipe_opt = Some(CleanedRecipe {
                    recipe_title: loaded_data.recipe_title.clone(),
                    ingredients: loaded_data.ingredients.clone(),
                    instructions: loaded_data.instructions.clone(),
                });
                initial_nutritional_profile_opt = Some(loaded_data.nutritional_profile.clone());
            }
            Err(e) => {
                println!("Failed to parse existing enriched file ({}). Will re-process if needed.", e);
            }
        }
    }

    let mut nutritional_index_opt: Option<NutritionalIndex> = None;
    let needs_fresh_processing = initial_cleaned_recipe_opt.is_none();
    let needs_optimization = !cli_args.optimization_targets.is_empty();

    // Initialize NutritionalIndex if we need to process from scratch OR if optimization is requested.
    if needs_fresh_processing || needs_optimization {
        println!("Initializing Nutritional Index (this may take a moment)...");
        nutritional_index_opt = Some(
            NutritionalIndex::new(Path::new(CIQUAL_CSV_PATH), API_KEY_ENV_VAR)
                .with_context(|| format!("Failed to initialize Nutritional Index with Ciqual data from '{}'", CIQUAL_CSV_PATH))?
        );
        println!("Nutritional Index initialized.");
    }
    
    let progress_callback = |message: String| { println!("{}", message); };

    let (mut current_cleaned_recipe, mut current_nutritional_profile) = 
        if let (Some(recipe), Some(profile)) = (initial_cleaned_recipe_opt, initial_nutritional_profile_opt) {
            // This block is entered if initial_cleaned_recipe_opt and initial_nutritional_profile_opt are Some
            println!("Using pre-loaded enriched recipe data as starting point.");
            (recipe, profile)
        } else {
            // This block is entered if loading failed or file didn't exist
            println!("Processing from raw recipe text...");
            let index = nutritional_index_opt.as_ref()
                .ok_or_else(|| anyhow!("NutritionalIndex not initialized for raw processing but is required."))?;

            let recipe_content = fs::read_to_string(&input_path)
                .await
                .with_context(|| format!("Failed to read recipe file '{}'", cli_args.recipe_file))?;
            println!("\nRecipe content read successfully. Sending to parser...");

            let parsed_recipe = parse_recipe_text(&recipe_content, API_KEY_ENV_VAR).await
                .with_context(|| "Recipe parsing failed")?;
            
            println!("\nSuccessfully parsed recipe. Now converting ingredients to grams...");
            
            let mut temp_cleaned_recipe = convert_ingredients_to_grams(&parsed_recipe, API_KEY_ENV_VAR, progress_callback.clone()).await
                .with_context(|| "Ingredient conversion to grams failed")?;
            
            println!("\nSuccessfully converted recipe ingredients to grams.");
            
            if let Err(e) = enrich_with_nutritional_info(&mut temp_cleaned_recipe, index, API_KEY_ENV_VAR, progress_callback.clone()).await {
                eprintln!("\nError enriching recipe with nutritional info: {}", e);
            }
            let profile = calculate_nutritional_profile(&temp_cleaned_recipe);
            (temp_cleaned_recipe, profile)
        };

    if needs_optimization {
        println!("\n--- Starting Recipe Optimization ---");
        let goals_map = cli_args.get_optimization_targets_map();
        let target_nutrition_per_100g = calculate_target_nutrition(
            &current_nutritional_profile.per_100g, 
            &goals_map,
        );
        println!("Target Nutritional Values (per 100g): {:#?}", target_nutrition_per_100g);
        
        let index_for_optim = nutritional_index_opt.as_ref()
            .ok_or_else(|| anyhow!("NutritionalIndex not initialized for optimization but is required."))?;

        match optimize_recipe(
            &current_cleaned_recipe,
            &current_nutritional_profile,
            &target_nutrition_per_100g,
            cli_args.max_iterations, 
            index_for_optim,
            API_KEY_ENV_VAR,
            progress_callback.clone(),
        ).await {
            Ok(optimized_recipe) => {
                println!("\n--- Optimization Complete ---");
                current_cleaned_recipe = optimized_recipe;
                current_nutritional_profile = calculate_nutritional_profile(&current_cleaned_recipe);
                println!("Optimized Recipe Title: {}", current_cleaned_recipe.recipe_title);
                println!("Optimized Nutritional Profile (Aggregated): {:#?}", current_nutritional_profile.aggregated); 
                println!("Optimized Nutritional Profile (Per 100g): {:#?}", current_nutritional_profile.per_100g);
                
                let optimized_output_data = EnrichedRecipeOutput {
                    recipe_title: current_cleaned_recipe.recipe_title.clone(),
                    ingredients: current_cleaned_recipe.ingredients.clone(),
                    instructions: current_cleaned_recipe.instructions.clone(),
                    nutritional_profile: current_nutritional_profile.clone(),
                };
                let optimized_json_output = serde_json::to_string_pretty(&optimized_output_data)
                    .with_context(|| "Failed to serialize optimized recipe to JSON")?;
                fs::write(&optimized_file_path, optimized_json_output)
                    .await
                    .with_context(|| format!("Failed to write optimized recipe to JSON file: {:?}", optimized_file_path))?;
                println!("\nOptimized recipe saved to '{}'", optimized_file_path.display());

            }
            Err(e) => {
                eprintln!("\nRecipe optimization failed: {}", e);
                println!("Proceeding with unoptimized recipe for final output (if it was processed).");
                // If optimization failed, we still have current_cleaned_recipe and current_nutritional_profile
                // which could be the initially loaded or processed one. We can save this to _enriched.json
                // if it hasn't been saved yet (e.g. if optimization was the only goal).
                if !enriched_file_path.exists() || needs_fresh_processing { // Save if it was freshly processed
                    let output_data = EnrichedRecipeOutput {
                        recipe_title: current_cleaned_recipe.recipe_title.clone(),
                        ingredients: current_cleaned_recipe.ingredients.clone(),
                        instructions: current_cleaned_recipe.instructions.clone(),
                        nutritional_profile: current_nutritional_profile.clone(),
                    };
                    let json_output = serde_json::to_string_pretty(&output_data)
                        .with_context(|| "Failed to serialize recipe to JSON after failed optimization")?;
                    fs::write(&enriched_file_path, json_output)
                        .await
                        .with_context(|| format!("Failed to write enriched recipe to JSON file after failed optimization: {:?}", enriched_file_path))?;
                    println!("\nUnoptimized (or initially processed) recipe saved to '{}'", enriched_file_path.display());
                }
            }
        }
    } else { // No optimization requested
        let output_data = EnrichedRecipeOutput {
            recipe_title: current_cleaned_recipe.recipe_title.clone(),
            ingredients: current_cleaned_recipe.ingredients.clone(),
            instructions: current_cleaned_recipe.instructions.clone(),
            nutritional_profile: current_nutritional_profile.clone(),
        };
        let json_output = serde_json::to_string_pretty(&output_data)
            .with_context(|| "Failed to serialize recipe to JSON")?;
        fs::write(&enriched_file_path, json_output)
            .await
            .with_context(|| format!("Failed to write enriched recipe to JSON file: {:?}", enriched_file_path))?;
        println!("\nEnriched recipe (unoptimized) saved to '{}'", enriched_file_path.display());
    }
    
    println!("\nSuccessfully processed recipe.");

    Ok(())
}
