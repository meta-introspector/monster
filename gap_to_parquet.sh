#!/usr/bin/env bash
# Convert GAP Monster data to Parquet using Rust

set -e

echo "📊 Converting Monster GAP data to Parquet"
echo "=========================================="
echo ""

# Create Rust program
cat > /tmp/gap_to_parquet.rs << 'RUST'
use std::fs;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct MonsterData {
    name: String,
    order: String,
    is_simple: bool,
    is_sporadic: bool,
    num_conjugacy_classes: i32,
    num_characters: i32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read JSON from GAP
    let json_str = fs::read_to_string("monster_gap_data.json")?;
    let data: MonsterData = serde_json::from_str(&json_str)?;
    
    println!("✅ Loaded Monster data:");
    println!("   Order: {}", data.order);
    println!("   Conjugacy classes: {}", data.num_conjugacy_classes);
    println!("   Characters: {}", data.num_characters);
    
    // Create Arrow record batch
    use arrow::array::{StringArray, BooleanArray, Int32Array};
    use arrow::datatypes::{Schema, Field, DataType};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;
    
    let schema = Schema::new(vec![
        Field::new("source", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("order", DataType::Utf8, false),
        Field::new("is_simple", DataType::Boolean, false),
        Field::new("is_sporadic", DataType::Boolean, false),
        Field::new("num_conjugacy_classes", DataType::Int32, false),
        Field::new("num_characters", DataType::Int32, false),
    ]);
    
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(StringArray::from(vec!["GAP"])),
            Arc::new(StringArray::from(vec![data.name.as_str()])),
            Arc::new(StringArray::from(vec![data.order.as_str()])),
            Arc::new(BooleanArray::from(vec![data.is_simple])),
            Arc::new(BooleanArray::from(vec![data.is_sporadic])),
            Arc::new(Int32Array::from(vec![data.num_conjugacy_classes])),
            Arc::new(Int32Array::from(vec![data.num_characters])),
        ],
    )?;
    
    // Write to Parquet
    let file = fs::File::create("monster_gap_data.parquet")?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    
    println!("\n✅ Saved to monster_gap_data.parquet");
    
    Ok(())
}
RUST

# Compile and run
echo "🔨 Compiling Rust converter..."
rustc /tmp/gap_to_parquet.rs \
    --extern serde=/path/to/serde \
    --extern serde_json=/path/to/serde_json \
    --extern arrow=/path/to/arrow \
    --extern parquet=/path/to/parquet \
    -o /tmp/gap_to_parquet 2>/dev/null || {
    
    echo "⚠️  Using cargo instead..."
    cd /tmp
    cargo init --bin gap_converter
    cd gap_converter
    
    cat > Cargo.toml << 'TOML'
[package]
name = "gap_converter"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
arrow = "53.0"
parquet = "53.0"
TOML
    
    cp /tmp/gap_to_parquet.rs src/main.rs
    cargo build --release
    cp target/release/gap_converter /tmp/gap_to_parquet
}

echo "✅ Compiled"
echo ""
echo "🚀 Running converter..."
/tmp/gap_to_parquet
