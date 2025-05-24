use anyhow::{Result, Context};
use csv::ReaderBuilder;
use std::path::Path;
use crate::recipe_converter::CiqualFoodItem; // Assuming CiqualFoodItem is in recipe_converter

// Define expected column headers
const NAME_COL: &str = "Name";
const KCAL_COL: &str = "kcal/100g";
const WATER_COL: &str = "Water (g/100g)";
const PROTEIN_COL: &str = "Protein (g/100g)";
const CARB_COL: &str = "Carbohydrate (g/100g)";
const FAT_COL: &str = "Fat (g/100g)";
const SUGARS_COL: &str = "Sugars (g/100g)";
const SAT_FAT_COL: &str = "FA saturated (g/100g)";
const SALT_COL: &str = "Salt (g/100g)";

fn parse_optional_f32(s: &str) -> Option<f32> {
    s.trim().parse::<f32>().ok()
}

pub fn load_ciqual_nutritional_data(csv_path: &Path) -> Result<Vec<CiqualFoodItem>> {
    if !csv_path.exists() {
        return Err(anyhow::anyhow!("Ciqual CSV file not found at: {:?}", csv_path));
    }

    let file = std::fs::File::open(csv_path)
        .with_context(|| format!("Failed to open Ciqual CSV file at {:?}", csv_path))?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let headers = rdr.headers()?.clone();
    
    // Get column indices
    let name_idx = headers.iter().position(|h| h == NAME_COL).ok_or_else(|| anyhow::anyhow!("Column '{}' not found", NAME_COL))?;
    let kcal_idx = headers.iter().position(|h| h == KCAL_COL).ok_or_else(|| anyhow::anyhow!("Column '{}' not found", KCAL_COL))?;
    let water_idx = headers.iter().position(|h| h == WATER_COL).ok_or_else(|| anyhow::anyhow!("Column '{}' not found", WATER_COL))?;
    let protein_idx = headers.iter().position(|h| h == PROTEIN_COL).ok_or_else(|| anyhow::anyhow!("Column '{}' not found", PROTEIN_COL))?;
    let carb_idx = headers.iter().position(|h| h == CARB_COL).ok_or_else(|| anyhow::anyhow!("Column '{}' not found", CARB_COL))?;
    let fat_idx = headers.iter().position(|h| h == FAT_COL).ok_or_else(|| anyhow::anyhow!("Column '{}' not found", FAT_COL))?;
    let sugars_idx = headers.iter().position(|h| h == SUGARS_COL).ok_or_else(|| anyhow::anyhow!("Column '{}' not found", SUGARS_COL))?;
    let sat_fat_idx = headers.iter().position(|h| h == SAT_FAT_COL).ok_or_else(|| anyhow::anyhow!("Column '{}' not found", SAT_FAT_COL))?;
    let salt_idx = headers.iter().position(|h| h == SALT_COL).ok_or_else(|| anyhow::anyhow!("Column '{}' not found", SALT_COL))?;

    let mut ciqual_data = Vec::new();
    for (row_index, result) in rdr.records().enumerate() {
        let record = result.with_context(|| format!("Failed to read record at row index {}", row_index))?;
        
        let name = record.get(name_idx).ok_or_else(|| anyhow::anyhow!("Missing name at row {}", row_index))?.trim().to_string();
        if name.is_empty() {
            // Skip rows with empty names, or handle as an error
            // eprintln!("Warning: Skipping row {} due to empty name.", row_index + 1); // +1 for header
            continue;
        }

        let item = CiqualFoodItem {
            name,
            original_row_index: row_index,
            kcal_per_100g: record.get(kcal_idx).and_then(|s| parse_optional_f32(s)),
            water_g_per_100g: record.get(water_idx).and_then(|s| parse_optional_f32(s)),
            protein_g_per_100g: record.get(protein_idx).and_then(|s| parse_optional_f32(s)),
            carbohydrate_g_per_100g: record.get(carb_idx).and_then(|s| parse_optional_f32(s)),
            fat_g_per_100g: record.get(fat_idx).and_then(|s| parse_optional_f32(s)),
            sugars_g_per_100g: record.get(sugars_idx).and_then(|s| parse_optional_f32(s)),
            fa_saturated_g_per_100g: record.get(sat_fat_idx).and_then(|s| parse_optional_f32(s)),
            salt_g_per_100g: record.get(salt_idx).and_then(|s| parse_optional_f32(s)),
        };
        ciqual_data.push(item);
    }

    if ciqual_data.is_empty() {
        return Err(anyhow::anyhow!("No valid Ciqual data loaded from {:?}", csv_path));
    }

    Ok(ciqual_data)
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv_file() -> Result<NamedTempFile> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "{},{},{},{},{},{},{},{},{}", 
                 NAME_COL, KCAL_COL, WATER_COL, PROTEIN_COL, CARB_COL, FAT_COL, SUGARS_COL, SAT_FAT_COL, SALT_COL)?;
        writeln!(file, "Apple,52,85.6,0.3,13.8,0.2,10.4,0.0,0.0")?;
        writeln!(file, "Banana,,75,1.1,22.8,0.3,12.2,0.1,0.0")?; // Missing kcal
        writeln!(file, "Carrot,41,88.3,0.9,9.6,0.2,4.7,0.0,0.07")?;
        writeln!(file, ",10,10,10,10,10,10,10,10")?; // Empty name
        writeln!(file, "InvalidNutrient,text,80,1,1,1,1,1,1")?; // Invalid kcal
        file.flush()?;
        Ok(file)
    }

    #[test]
    fn test_load_ciqual_nutritional_data_success() -> Result<()> {
        let file = create_test_csv_file()?;
        let data = load_ciqual_nutritional_data(file.path())?;
        
        assert_eq!(data.len(), 4); // "Apple", "Banana", "Carrot", "InvalidNutrient" (empty name row skipped)

        let apple = data.iter().find(|item| item.name == "Apple").unwrap();
        assert_eq!(apple.kcal_per_100g, Some(52.0));
        assert_eq!(apple.water_g_per_100g, Some(85.6));
        assert_eq!(apple.protein_g_per_100g, Some(0.3));

        let banana = data.iter().find(|item| item.name == "Banana").unwrap();
        assert_eq!(banana.kcal_per_100g, None); // kcal was missing
        assert_eq!(banana.water_g_per_100g, Some(75.0));

        let invalid_nutrient_item = data.iter().find(|item| item.name == "InvalidNutrient").unwrap();
        assert_eq!(invalid_nutrient_item.kcal_per_100g, None); // kcal was "text"
        assert_eq!(invalid_nutrient_item.water_g_per_100g, Some(80.0));

        Ok(())
    }

    #[test]
    fn test_load_ciqual_nutritional_data_missing_column() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        // Missing KCAL_COL
        writeln!(file, "{},{},{},{},{},{},{},{}", 
                 NAME_COL, WATER_COL, PROTEIN_COL, CARB_COL, FAT_COL, SUGARS_COL, SAT_FAT_COL, SALT_COL)?;
        writeln!(file, "Apple,85.6,0.3,13.8,0.2,10.4,0.0,0.0")?;
        file.flush()?;

        let result = load_ciqual_nutritional_data(file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains(&format!("Column '{}' not found", KCAL_COL)));
        Ok(())
    }

    #[test]
    fn test_load_ciqual_nutritional_data_empty_file_with_headers() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "{},{},{},{},{},{},{},{},{}", 
                 NAME_COL, KCAL_COL, WATER_COL, PROTEIN_COL, CARB_COL, FAT_COL, SUGARS_COL, SAT_FAT_COL, SALT_COL)?;
        file.flush()?;

        let result = load_ciqual_nutritional_data(file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No valid Ciqual data loaded"));
        Ok(())
    }
    
    #[test]
    fn test_load_ciqual_nutritional_data_file_not_found() {
        let path = Path::new("this_file_does_not_exist.csv");
        let result = load_ciqual_nutritional_data(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Ciqual CSV file not found"));
    }
}
