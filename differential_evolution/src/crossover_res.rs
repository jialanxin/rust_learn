use crate::mutation_force::MutationForce;
use crate::particle::Particle;
use rand::Rng;
#[derive(Debug)]
pub struct CrossoverResult {
    pub crossover_result: Vec<Option<f64>>,
}
impl CrossoverResult {
    pub fn new(dims: usize) -> Self {
        CrossoverResult {
            crossover_result: vec![None; dims],
        }
    }
    pub fn calc_crossover_res(
        &mut self,
        cross_probability: f64,
        particle: &Particle,
        mutation_force: &MutationForce,
        force_change_index: usize,
    ) {
        let r = rand::thread_rng().gen::<f64>();
        for (j, u) in self.crossover_result.iter_mut().enumerate() {
            if r <= cross_probability || j == force_change_index {
                *u = Some(
                    mutation_force
                        .mutation_force
                        .as_ref()
                        .unwrap()
                        .get(j)
                        .unwrap()
                        .clone(),
                )
            } else {
                *u = Some(particle.position.get(j).unwrap().clone())
            }
        }
    }
}
