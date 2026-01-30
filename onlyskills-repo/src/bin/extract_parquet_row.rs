// Extract Row from Parquet
use polars::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: extract_parquet_row <parquet_file> <row_index>");
        std::process::exit(1);
    }
    
    let parquet_file = &args[1];
    let row_idx: usize = args[2].parse().expect("Invalid row index");
    
    // Read parquet file
    let df = LazyFrame::scan_parquet(parquet_file, Default::default())
        .expect("Failed to scan parquet")
        .collect()
        .expect("Failed to collect");
    
    // Extract row
    if row_idx >= df.height() {
        eprintln!("Row index out of bounds");
        std::process::exit(1);
    }
    
    // Print row as JSON
    println!("{{");
    for (i, col_name) in df.get_column_names().iter().enumerate() {
        let col = df.column(col_name).expect("Column not found");
        let val = format!("{:?}", col.get(row_idx).expect("Failed to get value"));
        
        if i > 0 {
            println!(",");
        }
        print!("  \"{}\": {}", col_name, val);
    }
    println!("\n}}");
}
