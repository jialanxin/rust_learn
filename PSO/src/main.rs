mod lib;
use lib::classic_pso;
use std::time::Instant;

fn main() {
    let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x
        .iter()
        .map(|x| (-x / 1.0).exp() * 1.0 + (-x / 2.0).exp() * 2.0)
        .collect();
    let position_max: Vec<f64> = vec![3.0, 3.0, 3.0, 3.0];
    let position_min: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0];
    let start = Instant::now();
    let (loss, best_position) =
        classic_pso(position_max, position_min, x, y, 100000, 1.0, 2.0, 2.0, 100).unwrap();
    let duration = start.elapsed();
    println!("{}", loss);
    println!("{:?}", best_position);
    println!("Time elapsed in expensive_function() is: {:?}", duration);
}
