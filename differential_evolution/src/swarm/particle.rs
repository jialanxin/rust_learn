use super::crossover_res::CrossoverResult;
use crate::evaluate;
use rand::Rng;
#[derive(Debug)]
pub struct Particle {
    pub position: Vec<f64>,
}
impl Particle {
    pub fn new(position_max: &Vec<f64>, position_min: &Vec<f64>) -> Self {
        let position_range: Vec<f64> = position_max
            .iter()
            .zip(position_min.iter())
            .map(|(a, b)| a - b)
            .collect();
        let mut rng = rand::thread_rng();
        let position: Vec<f64> = position_range
            .iter()
            .map(|a| rng.gen::<f64>() * a)
            .zip(position_min.iter())
            .map(|(a, b)| a + b)
            .collect();
        Particle { position: position }
    }
    pub fn select(
        &mut self,
        x_data: &Vec<f64>,
        y_data: &Vec<f64>,
        crossover_res: &CrossoverResult,
    ) {
        let origin = evaluate(&self.position, x_data, y_data);
        let new_position = crossover_res
            .crossover_result
            .iter()
            .map(|x| x.unwrap())
            .collect::<Vec<f64>>();
        let new_fit = evaluate(&new_position, x_data, y_data);
        if new_fit <= origin {
            self.position = new_position;
        }
    }
}
