use clap::Parser;
use std::str::FromStr;
use std::collections::HashMap; // To store parsed optimization targets

// Define an enum for the nutrients we can target for percentage change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizableNutrient {
    Carb,
    Fat,
    Protein,
    // Kcal is removed as a direct percentage target for --optimize.
    // It will be an outcome of macronutrient changes.
    // Add Sugars, Fiber etc. as needed in the future
}

impl FromStr for OptimizableNutrient {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "carb" | "carbohydrate" | "carbohydrates" => Ok(OptimizableNutrient::Carb),
            "fat" | "fats" => Ok(OptimizableNutrient::Fat),
            "protein" | "proteins" => Ok(OptimizableNutrient::Protein),
            _ => Err(format!("Unknown nutrient for --optimize: '{}'. Supported: carb, fat, protein.", s)),
        }
    }
}

// Custom parser for the <nutrient>:<percentage_change> format
fn parse_optimization_target(s: &str) -> Result<(OptimizableNutrient, f32), String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(format!(
            "Invalid format for optimization target: '{}'. Expected <nutrient>:<percentage_change>",
            s
        ));
    }

    let nutrient = OptimizableNutrient::from_str(parts[0])?;
    let percentage = parts[1]
        .parse::<f32>()
        .map_err(|e| format!("Invalid percentage value '{}': {}", parts[1], e))?;

    Ok((nutrient, percentage))
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the recipe text file
    #[arg(short, long)]
    pub recipe_file: String,

    /// Optimization targets for macronutrients (carb, fat, protein), can be specified multiple times.
    /// Format: <nutrient>:<percentage_change>
    /// Example: --optimize carb:-10 --optimize protein:+20
    /// Supported nutrients: carb, fat, protein.
    /// Kcal will be affected indirectly by these changes.
    /// Percentage change: e.g., -10 for 10% reduction, +20 for 20% increase.
    #[arg(long = "optimize", value_parser = parse_optimization_target, action = clap::ArgAction::Append)]
    pub optimization_targets: Vec<(OptimizableNutrient, f32)>,

    /// Maximum number of optimization iterations
    #[arg(long, default_value_t = 10)]
    pub max_iterations: u32,
}

impl Cli {
    /// Helper to get optimization targets as a HashMap for easier lookup
    pub fn get_optimization_targets_map(&self) -> HashMap<OptimizableNutrient, f32> {
        self.optimization_targets.iter().cloned().collect()
    }
}

pub fn parse_args() -> Cli {
    Cli::parse()
}
