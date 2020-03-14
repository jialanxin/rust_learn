mod crossover_res;
mod mutation_force;
mod particle;
mod swarm;
use self::swarm::Swarm;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub fn evaluate(params: &Vec<f64>, x_data: &Vec<f64>, y_data: &Vec<f64>) -> f64 {
    (params[0] * x_data[0] - y_data[0]).powf(2.0)
}
#[pyfunction]
fn de(position_max:Vec<f64>,position_min:Vec<f64>,x_data:Vec<f64>,y_data:Vec<f64>,num_of_particles:usize,differential_weight:f64,crossover_probability:f64,steps:usize)->PyResult<(f64,Vec<f64>)>{
    let mut a_swarm = Swarm::new(num_of_particles, &position_max, &position_min);
    for _ in 0..steps{
        a_swarm.evolution(differential_weight, crossover_probability, &x_data, &y_data);
    }
    let (best_loss,best_position) = a_swarm.result(&x_data, &y_data);
    Ok((best_loss,best_position))
}
#[pymodule]
fn differential_evolution(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(de))?;

    Ok(())
}
