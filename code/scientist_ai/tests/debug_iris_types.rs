/// Debug test: Check how Rust detects Iris variable types
use polars::prelude::*;
use scientist_ai::inference::get_variable_type;

#[test]
fn test_iris_variable_types() {
    println!("\n=== Debugging: Iris Variable Type Detection ===\n");

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("data/iris.csv".into()))
        .expect("Failed to create CSV reader")
        .finish()
        .expect("Failed to read iris.csv");

    println!("Dataset info:");
    println!("  Rows: {}", df.height());
    println!("  Columns: {}\n", df.width());

    for col_name in df.get_column_names() {
        let col = df.column(col_name).unwrap();
        let dtype = col.dtype();
        let unique_count = col.n_unique().unwrap();

        let var_type = get_variable_type(&df, col_name)
            .expect(&format!("Failed to get type for {}", col_name));

        println!("{:15} | dtype: {:?} | unique: {:3} | detected: {:?}",
                 col_name, dtype, unique_count, var_type);

        // Show sample values for Species
        if col_name == "Species" {
            println!("  Sample values: {:?}", col.head(Some(5)));
        }
    }

    println!("\nExpected:");
    println!("  SepalLength  -> Continuous (f64, many unique values)");
    println!("  SepalWidth   -> Continuous (f64, many unique values)");
    println!("  PetalLength  -> Continuous (f64, many unique values)");
    println!("  PetalWidth   -> Continuous (f64, many unique values)");
    println!("  Species      -> Categorical (3 unique: setosa, versicolor, virginica)");
}
