use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the recipe text file
    #[arg(short, long)]
    pub recipe_file: String,
}

pub fn parse_args() -> Cli {
    Cli::parse()
}
