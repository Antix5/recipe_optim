use crate::recipe_aggregator::NutritionalSummary;
use crate::optim::targets::TargetNutritionalValues;

/// Calculates the Mean Squared Error (MSE) between the nutritional profile of a recipe
/// (per 100g) and the target nutritional values (per 100g).
///
/// The MSE is calculated for key macronutrients: protein, carbohydrates, and fat.
/// Kcal can also be included if desired, though it's derived.
/// Only fields present in both the profile and target are included in the MSE calculation.
///
/// # Arguments
/// * `current_profile_per_100g`: The nutritional summary of the current recipe, per 100g.
/// * `target_values_per_100g`: The target nutritional values, per 100g.
///
/// # Returns
/// The calculated MSE as an f32. Returns 0.0 if no common fields with values are found.
pub fn calculate_mse(
    current_profile_per_100g: &NutritionalSummary,
    target_values_per_100g: &TargetNutritionalValues,
) -> f32 {
    let mut squared_error_sum = 0.0;
    let mut count = 0;

    // Protein
    if let (Some(current_p), Some(target_p)) = (current_profile_per_100g.protein_g, target_values_per_100g.protein_g) {
        squared_error_sum += (current_p - target_p).powi(2);
        count += 1;
    }

    // Carbohydrates
    if let (Some(current_c), Some(target_c)) = (current_profile_per_100g.carbohydrate_g, target_values_per_100g.carbohydrate_g) {
        squared_error_sum += (current_c - target_c).powi(2);
        count += 1;
    }

    // Fat
    if let (Some(current_f), Some(target_f)) = (current_profile_per_100g.fat_g, target_values_per_100g.fat_g) {
        squared_error_sum += (current_f - target_f).powi(2);
        count += 1;
    }

    // Kcal (optional, as it's derived, but can be part of the target)
    if let (Some(current_kcal), Some(target_kcal)) = (current_profile_per_100g.kcal, target_values_per_100g.kcal) {
        // Kcal values can be much larger, so consider normalizing or weighting if it dominates MSE
        // For now, direct MSE.
        squared_error_sum += (current_kcal - target_kcal).powi(2) / 100.0; // Simple scaling for kcal
        count += 1;
    }
    
    // Add other nutrients if they become primary targets, e.g., sugars, fiber, etc.
    // if let (Some(current_s), Some(target_s)) = (current_profile_per_100g.sugars_g, target_values_per_100g.sugars_g) {
    //     squared_error_sum += (current_s - target_s).powi(2);
    //     count += 1;
    // }

    if count == 0 {
        0.0 // Or perhaps f32::MAX if no common targets could be evaluated, indicating a problem.
            // For now, 0.0 means no error if no targets are set for these fields.
    } else {
        squared_error_sum / count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe_aggregator::NutritionalSummary;
    use crate::optim::targets::TargetNutritionalValues;

    #[test]
    fn test_calculate_mse_perfect_match() {
        let profile = NutritionalSummary {
            kcal: Some(200.0),
            protein_g: Some(20.0),
            carbohydrate_g: Some(15.0),
            fat_g: Some(5.0),
            ..Default::default()
        };
        let target = TargetNutritionalValues {
            kcal: Some(200.0),
            protein_g: Some(20.0),
            carbohydrate_g: Some(15.0),
            fat_g: Some(5.0),
            ..Default::default()
        };
        assert_eq!(calculate_mse(&profile, &target), 0.0);
    }

    #[test]
    fn test_calculate_mse_some_diff() {
        let profile = NutritionalSummary {
            kcal: Some(210.0), // diff 10, scaled sq_err = (100/100) = 1
            protein_g: Some(18.0), // diff -2, sq_err = 4
            carbohydrate_g: Some(20.0), // diff 5, sq_err = 25
            fat_g: Some(6.0), // diff 1, sq_err = 1
            ..Default::default()
        };
        let target = TargetNutritionalValues {
            kcal: Some(200.0),
            protein_g: Some(20.0),
            carbohydrate_g: Some(15.0),
            fat_g: Some(5.0),
            ..Default::default()
        };
        // Sum of squared errors = 1 (kcal scaled) + 4 + 25 + 1 = 31
        // Count = 4
        // MSE = 31 / 4 = 7.75
        assert_eq!(calculate_mse(&profile, &target), 7.75);
    }

    #[test]
    fn test_calculate_mse_missing_target_fields() {
        let profile = NutritionalSummary {
            kcal: Some(200.0),
            protein_g: Some(20.0),
            carbohydrate_g: Some(15.0),
            fat_g: Some(5.0),
            ..Default::default()
        };
        let target = TargetNutritionalValues {
            kcal: None, // Kcal target missing
            protein_g: Some(20.0), // Match
            carbohydrate_g: Some(10.0), // Diff 5, sq_err = 25
            fat_g: None, // Fat target missing
            ..Default::default()
        };
        // Sum of squared errors = 0 (protein) + 25 (carbs) = 25
        // Count = 2 (protein, carbs)
        // MSE = 25 / 2 = 12.5
        assert_eq!(calculate_mse(&profile, &target), 12.5);
    }

    #[test]
    fn test_calculate_mse_missing_profile_fields() {
        let profile = NutritionalSummary {
            kcal: None,
            protein_g: Some(20.0), // Match
            carbohydrate_g: None,
            fat_g: Some(5.0), // Diff -2, sq_err = 4
            ..Default::default()
        };
        let target = TargetNutritionalValues {
            kcal: Some(200.0),
            protein_g: Some(20.0),
            carbohydrate_g: Some(15.0),
            fat_g: Some(7.0), 
            ..Default::default()
        };
        // Sum of squared errors = 0 (protein) + 4 (fat) = 4
        // Count = 2 (protein, fat)
        // MSE = 4 / 2 = 2.0
        assert_eq!(calculate_mse(&profile, &target), 2.0);
    }

    #[test]
    fn test_calculate_mse_no_common_fields() {
        let profile = NutritionalSummary {
            sugars_g: Some(10.0), // Only sugars in profile
            ..Default::default()
        };
        let target = TargetNutritionalValues {
            protein_g: Some(20.0), // Only protein in target
            ..Default::default()
        };
        // No common fields for primary MSE calculation (kcal, P, C, F)
        assert_eq!(calculate_mse(&profile, &target), 0.0);
    }
}
