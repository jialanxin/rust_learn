mod lib;
use lib::classic_pso;
use std::time::Instant;

fn main() {
    let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x
        .iter()
        .map(|x| 100000.0/(10000.0+x.powi(2)))
        .collect();
    let position_max: Vec<f64> = vec![200000.0, 20000.0];
    let position_min: Vec<f64> = vec![0.0, 0.0];
    let start = Instant::now();
    let (loss, best_position, mean, stdvar) = classic_pso(
        position_max,
        position_min,
        x,
        y,
        100_000,
        1.0,
        2.0,
        2.0,
        200,
    )
    .unwrap();
    let duration = start.elapsed();
    println!("{}", loss);
    println!("{:?}", best_position);
    println!("{:?}", mean);
    println!("{:?}", stdvar);
    println!("Time elapsed in expensive_function() is: {:?}", duration);
}
