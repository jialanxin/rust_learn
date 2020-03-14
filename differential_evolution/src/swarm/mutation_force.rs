use super::particle::Particle;
#[derive(Debug)]
pub struct MutationForce {
    pub mutation_force: Option<Vec<f64>>,
}
impl MutationForce {
    pub fn new() -> Self {
        MutationForce {
            mutation_force: None,
        }
    }
    pub fn calc_mutation_force(
        &mut self,
        differential_weight: f64,
        particle1: &Particle,
        particle2: &Particle,
        particle3: &Particle,
    ) {
        self.mutation_force = Some(
            particle1
                .position
                .iter()
                .zip(particle2.position.iter())
                .map(|(a, b)| (a - b) * differential_weight)
                .zip(particle3.position.iter())
                .map(|(a, b)| a + b)
                .collect(),
        )
    }
}
