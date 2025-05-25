# Recipe Optimization

This module provides functionality for optimizing recipes based on nutritional information and user preferences. It includes features for modifying ingredient quantities, substituting ingredients, and adjusting recipes to meet specific dietary goals.

## Key Components

- **Recipe Representation**: The core data structures for representing recipes, ingredients, and their nutritional information.
- **Optimization Algorithms**: Using an agentic loop to reduce the MSE loss

```bash
cargo run -- --recipe-file my_recipe.txt --optimize carb:-10 --optimize fat:-20
```