use rand::prelude::*;

fn main() {
    println!("Testing rand::thread_rng() randomness");
    println!("If this prints different values each run, RNG is working\n");

    let mut rng = rand::thread_rng();

    println!("10 random floats [0, 1):");
    for i in 0..10 {
        let val: f64 = rng.gen_range(0.0..1.0);
        println!("  {}: {}", i, val);
    }

    println!("\n10 random integers [0, 100):");
    for i in 0..10 {
        let val: u32 = rng.gen_range(0..100);
        println!("  {}: {}", i, val);
    }

    println!("\n10 random bools:");
    for i in 0..10 {
        let val: bool = rng.gen_bool(0.5);
        println!("  {}: {}", i, val);
    }

    println!("\nIf these values are different each time you run this,");
    println!("then rand::thread_rng() is working correctly.");
}
