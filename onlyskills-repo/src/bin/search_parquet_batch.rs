// Search Parquet Batch for Zero Ontology
use polars::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: search_parquet_batch <file_list> <pattern1> [pattern2...]");
        std::process::exit(1);
    }
    
    let file_list = &args[1];
    let patterns: Vec<&str> = args[2..].iter().map(|s| s.as_str()).collect();
    
    // Read file list
    let file = File::open(file_list).expect("Failed to open file list");
    let reader = BufReader::new(file);
    
    for line in reader.lines() {
        let parquet_file = line.expect("Failed to read line");
        search_parquet_file(&parquet_file, &patterns);
    }
}

fn search_parquet_file(path: &str, patterns: &[&str]) {
    // Read parquet file
    let df = match LazyFrame::scan_parquet(path, Default::default()) {
        Ok(lf) => match lf.collect() {
            Ok(df) => df,
            Err(_) => return,
        },
        Err(_) => return,
    };
    
    // Search all string columns
    for col_name in df.get_column_names() {
        if let Ok(col) = df.column(col_name) {
            if let Ok(str_col) = col.str() {
                for (row_idx, opt_val) in str_col.into_iter().enumerate() {
                    if let Some(val) = opt_val {
                        for pattern in patterns {
                            if val.contains(pattern) {
                                println!("{}:{}:{}:{}", path, row_idx, col_name, pattern);
                            }
                        }
                    }
                }
            }
        }
    }
}
