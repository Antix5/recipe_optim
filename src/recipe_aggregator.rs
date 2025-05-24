use serde::{Deserialize, Serialize};
use crate::recipe_converter::{CleanedRecipe, CleanedIngredient, CalculatedNutritionalInfo};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct NutritionalSummary { // Renamed for clarity, represents absolute values
    pub kcal: Option<f32>,
    pub water_g: Option<f32>,
    pub protein_g: Option<f32>,
    pub carbohydrate_g: Option<f32>,
    pub fat_g: Option<f32>,
    pub sugars_g: Option<f32>,
    pub fa_saturated_g: Option<f32>,
    pub salt_g: Option<f32>,
    // Add other fields if CiqualFoodItem/CalculatedNutritionalInfo has more
}

// This struct will hold both aggregated and per 100g normalized values
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RecipeNutritionalProfile {
    pub total_calculated_mass_g: Option<f32>,
    pub aggregated: NutritionalSummary,
    pub per_100g: NutritionalSummary, // Same fields, but values normalized per 100g
}


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EnrichedRecipeOutput {
    pub recipe_title: String,
    pub ingredients: Vec<CleanedIngredient>,
    pub instructions: Vec<String>,
    pub nutritional_profile: RecipeNutritionalProfile, // Changed from aggregated_nutrition
}

// Function to perform the aggregation and normalization
pub fn calculate_nutritional_profile(cleaned_recipe: &CleanedRecipe) -> RecipeNutritionalProfile {
    let mut aggregated_nutrition = NutritionalSummary::default();
    let mut total_mass_g = 0.0_f32;

    for ingredient in &cleaned_recipe.ingredients {
        if let (Some(grams), Some(nut_info)) = (ingredient.quantity_grams, &ingredient.nutritional_info) {
            if grams > 0.0 {
                total_mass_g += grams;
                macro_rules! add_optional {
                    ($field:ident) => {
                        if let Some(value) = nut_info.$field {
                            aggregated_nutrition.$field = Some(aggregated_nutrition.$field.unwrap_or(0.0) + value);
                        }
                    };
                }
                add_optional!(kcal);
                add_optional!(water_g);
                add_optional!(protein_g);
                add_optional!(carbohydrate_g);
                add_optional!(fat_g);
                add_optional!(sugars_g);
                add_optional!(fa_saturated_g);
                add_optional!(salt_g);
            }
        }
    }

    let mut per_100g_nutrition = NutritionalSummary::default();
    if total_mass_g > 0.0 {
        let scale_factor = 100.0 / total_mass_g;
        macro_rules! normalize_optional {
            ($field:ident) => {
                if let Some(agg_value) = aggregated_nutrition.$field {
                    per_100g_nutrition.$field = Some(agg_value * scale_factor);
                }
            };
        }
        normalize_optional!(kcal);
        normalize_optional!(water_g);
        normalize_optional!(protein_g);
        normalize_optional!(carbohydrate_g);
        normalize_optional!(fat_g);
        normalize_optional!(sugars_g);
        normalize_optional!(fa_saturated_g);
        normalize_optional!(salt_g);
    }

    RecipeNutritionalProfile {
        total_calculated_mass_g: if total_mass_g > 0.0 { Some(total_mass_g) } else { None },
        aggregated: aggregated_nutrition,
        per_100g: per_100g_nutrition,
    }
}
