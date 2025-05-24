use crate::cli::OptimizableNutrient;
use crate::recipe_aggregator::NutritionalSummary; // Using the per-100g or aggregated summary
use std::collections::HashMap;

// This struct will hold the desired absolute nutrient values after percentage changes.
// It mirrors NutritionalSummary for direct comparison.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TargetNutritionalValues {
    pub kcal: Option<f32>,
    pub water_g: Option<f32>, // Water might not be a direct target but included for completeness
    pub protein_g: Option<f32>,
    pub carbohydrate_g: Option<f32>,
    pub fat_g: Option<f32>,
    pub sugars_g: Option<f32>,
    pub fa_saturated_g: Option<f32>,
    pub salt_g: Option<f32>,
    // Add other fields if NutritionalSummary has more
}

/// Calculates the target nutritional values based on an initial profile and percentage changes.
///
/// # Arguments
/// * `initial_profile_per_100g`: The nutritional summary (e.g., per 100g) of the original recipe.
/// * `optimization_goals`: A map of nutrients to their desired percentage changes (e.g., Carb -> -10.0 for 10% reduction).
///
/// # Returns
/// A `TargetNutritionalValues` struct with the calculated absolute target values.
pub fn calculate_target_nutrition(
    initial_profile_per_100g: &NutritionalSummary,
    optimization_goals: &HashMap<OptimizableNutrient, f32>,
) -> TargetNutritionalValues {
    let mut target_values = TargetNutritionalValues {
        // Initialize with initial values, then adjust based on goals
        kcal: initial_profile_per_100g.kcal,
        water_g: initial_profile_per_100g.water_g,
        protein_g: initial_profile_per_100g.protein_g,
        carbohydrate_g: initial_profile_per_100g.carbohydrate_g,
        fat_g: initial_profile_per_100g.fat_g,
        sugars_g: initial_profile_per_100g.sugars_g,
        fa_saturated_g: initial_profile_per_100g.fa_saturated_g,
        salt_g: initial_profile_per_100g.salt_g,
    };

    for (nutrient, percentage_change) in optimization_goals {
        let multiplier = 1.0 + (percentage_change / 100.0);
        match nutrient {
            // Kcal is no longer a direct percentage target here.
            // It will be recalculated based on the modified macronutrients later if needed,
            // or the LLM will try to achieve a new kcal profile by adjusting macros.
            // For now, target_values.kcal retains the initial_profile_per_100g.kcal.
            OptimizableNutrient::Protein => {
                if let Some(val) = target_values.protein_g {
                    target_values.protein_g = Some(val * multiplier);
                }
            }
            OptimizableNutrient::Carb => {
                if let Some(val) = target_values.carbohydrate_g {
                    target_values.carbohydrate_g = Some(val * multiplier);
                }
            }
            OptimizableNutrient::Fat => {
                if let Some(val) = target_values.fat_g {
                    target_values.fat_g = Some(val * multiplier);
                }
            }
            // Note: Add cases for Sugars, Saturated Fat, Fiber etc. if they become optimizable
            // and are part of OptimizableNutrient and NutritionalSummary/TargetNutritionalValues.
        }
    }
    // After applying percentage changes to macros, we could recalculate an estimated Kcal target
    // using Atwater factors (Protein: 4 kcal/g, Carb: 4 kcal/g, Fat: 9 kcal/g).
    // However, for now, target_values.kcal will reflect the original kcal,
    // and the LLM's goal will be to hit the target macros, which will implicitly define the new kcal.
    // If a specific kcal target is desired *independently*, it would need a different CLI mechanism.

    // Recalculate kcal based on modified macros (optional, but good for consistency if macros are primary targets)
    let mut new_kcal = 0.0;
    let mut has_macros = false;
    if let Some(p) = target_values.protein_g { new_kcal += p * 4.0; has_macros = true; }
    if let Some(c) = target_values.carbohydrate_g { new_kcal += c * 4.0; has_macros = true; }
    if let Some(f) = target_values.fat_g { new_kcal += f * 9.0; has_macros = true; }

    if has_macros {
        target_values.kcal = Some(new_kcal);
    }
    // If no macros were present in the initial profile, kcal remains as it was (possibly None).

    target_values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_target_nutrition_reduce_carb() {
        let initial = NutritionalSummary {
            kcal: Some(200.0),
            protein_g: Some(10.0),
            carbohydrate_g: Some(30.0),
            fat_g: Some(5.0),
            ..Default::default()
        };
        let mut goals = HashMap::new();
        goals.insert(OptimizableNutrient::Carb, -10.0); // Reduce carbs by 10%

        let target = calculate_target_nutrition(&initial, &goals);
        assert_eq!(target.kcal, Some(200.0));
        assert_eq!(target.protein_g, Some(10.0));
        assert_eq!(target.carbohydrate_g, Some(27.0)); // 30 * 0.9 = 27
        assert_eq!(target.fat_g, Some(5.0));
    }

    #[test]
    fn test_calculate_target_nutrition_increase_protein_reduce_fat() {
        let initial = NutritionalSummary {
            kcal: Some(500.0),
            protein_g: Some(20.0),
            carbohydrate_g: Some(50.0),
            fat_g: Some(20.0),
            ..Default::default()
        };
        let mut goals = HashMap::new();
        goals.insert(OptimizableNutrient::Protein, 25.0); // Increase protein by 25%
        goals.insert(OptimizableNutrient::Fat, -50.0);   // Reduce fat by 50%

        let target = calculate_target_nutrition(&initial, &goals);
        assert_eq!(target.kcal, Some(500.0)); // Kcal not targeted directly
        assert_eq!(target.protein_g, Some(25.0));    // 20 * 1.25 = 25
        assert_eq!(target.carbohydrate_g, Some(50.0));
        assert_eq!(target.fat_g, Some(10.0));        // 20 * 0.5 = 10
    }

     #[test]
    fn test_calculate_target_nutrition_no_change() {
        let initial = NutritionalSummary {
            kcal: Some(100.0),
            protein_g: Some(10.0),
            ..Default::default()
        };
        let goals = HashMap::new(); // No optimization goals

        let target = calculate_target_nutrition(&initial, &goals);
        assert_eq!(target.kcal, Some(100.0));
        assert_eq!(target.protein_g, Some(10.0));
    }

    #[test]
    fn test_calculate_target_nutrition_kcal_recalculation() {
        let initial = NutritionalSummary {
            kcal: Some(440.0), // 20*4 + 50*4 + 20*9 = 80 + 200 + 180 = 460 - let's assume initial was slightly off or had other minor contributors
            protein_g: Some(20.0),
            carbohydrate_g: Some(50.0),
            fat_g: Some(20.0),
            ..Default::default()
        };
        let mut goals = HashMap::new();
        goals.insert(OptimizableNutrient::Protein, 25.0); // Target P: 25g
        goals.insert(OptimizableNutrient::Fat, -50.0);   // Target F: 10g
                                                         // Carbs remain 50g

        let target = calculate_target_nutrition(&initial, &goals);
        // Expected Kcal: 25*4 (P) + 50*4 (C) + 10*9 (F) = 100 + 200 + 90 = 390
        assert_eq!(target.protein_g, Some(25.0));
        assert_eq!(target.carbohydrate_g, Some(50.0)); // Unchanged
        assert_eq!(target.fat_g, Some(10.0));
        assert_eq!(target.kcal, Some(390.0));
    }

    #[test]
    fn test_kcal_unchanged_if_no_macros_initially() {
         let initial = NutritionalSummary {
            kcal: Some(100.0), // Only kcal, no macros
            protein_g: None,
            carbohydrate_g: None,
            fat_g: None,
            ..Default::default()
        };
        let mut goals = HashMap::new();
        goals.insert(OptimizableNutrient::Protein, 10.0); // This goal won't apply as initial protein is None

        let target = calculate_target_nutrition(&initial, &goals);
        assert_eq!(target.kcal, Some(100.0)); // Kcal should remain as initial, not become 0 or None due to no macros
        assert_eq!(target.protein_g, None); // Still None
    }
}
