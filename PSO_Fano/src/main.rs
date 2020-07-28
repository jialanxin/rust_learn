mod lib;
use lib::classic_pso;
use std::time::Instant;

fn main() {
    let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x
        .iter()
        .map(|x| (x-286.0)/5.0).map(|s|120.0*(1.0+s/1000.0).powi(2)/(1.0+s.powi(2)))
        .collect();
    let position_max: Vec<f64> = vec![290.0, 1000.0,5.0,120.0];
    let position_min: Vec<f64> = vec![285.0, 1000.0,0.0,0.0];

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
