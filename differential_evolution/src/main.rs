mod crossover_res;
mod mutation_force;
mod particle;
mod swarm;
use self::swarm::Swarm;
pub fn evaluate(params: &Vec<f64>, x_data: &Vec<f64>, y_data: &Vec<f64>) -> f64 {
    (params[0] * x_data[0] - y_data[0]).powf(2.0)
}
fn main() {
    let position_max = vec![2.];
    let position_min = vec![0.];
    let x_data = vec![1.0];
    let y_data = vec![1.0];
    let mut a_swarm = Swarm::new(20, &position_max, &position_min);
    for _ in 0..10 {
        a_swarm.evolution(0.5, 0.5, &x_data, &y_data);
    }
    a_swarm.result(&x_data, &y_data)
}
