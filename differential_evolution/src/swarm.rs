mod crossover_res;
mod mutation_force;
mod particle;
use self::crossover_res::CrossoverResult;
use super::evaluate;
use self::mutation_force::MutationForce;
use self::particle::Particle;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
#[derive(Debug)]
pub struct Swarm {
    pub particles: Vec<Particle>,
    mutation_force_list: Vec<MutationForce>,
    crossover_res_list: Vec<CrossoverResult>,
    number_of_particles: usize,
}
impl Swarm {
    pub fn new(number_of_particles: usize, up_bound: &Vec<f64>, down_bound: &Vec<f64>) -> Self {
        let dims = up_bound.len();
        let mut particles: Vec<Particle> = Vec::new();
        let mut mutation_force_list: Vec<MutationForce> = Vec::new();
        let mut crossover_res_list: Vec<CrossoverResult> = Vec::new();
        particles.reserve(number_of_particles);
        mutation_force_list.reserve(number_of_particles);
        crossover_res_list.reserve(number_of_particles);
        for _ in 0..number_of_particles {
            particles.push(Particle::new(up_bound, down_bound));
            mutation_force_list.push(MutationForce::new());
            crossover_res_list.push(CrossoverResult::new(dims));
        }
        Swarm {
            particles: particles,
            mutation_force_list: mutation_force_list,
            crossover_res_list: crossover_res_list,
            number_of_particles: number_of_particles,
        }
    }
    fn mutation(&mut self, differential_weight: f64) {
        let particles = &self.particles;
        self.mutation_force_list.par_iter_mut().for_each(|x| {
            let mut rng = thread_rng();
            let particle1 = particles.choose(&mut rng).unwrap();
            let particle2 = particles.choose(&mut rng).unwrap();
            let particle3 = particles.choose(&mut rng).unwrap();
            x.calc_mutation_force(differential_weight, particle1, particle2, particle3)
        })
    }
    fn croseeover(&mut self, crossover_probability: f64) {
        let particles = &mut self.particles;
        let mutation_force_list = &self.mutation_force_list;
        let mut rng = thread_rng();
        let nums: Vec<usize> = (0..self.number_of_particles).collect();
        let force_change_index: usize = *nums.choose(&mut rng).unwrap();
        self.crossover_res_list
            .par_iter_mut()
            .zip(
                particles
                    .par_iter()
                    .zip(mutation_force_list)
                    .collect::<Vec<(&Particle, &MutationForce)>>(),
            )
            .for_each(|(c, (p, m))| {
                c.calc_crossover_res(crossover_probability, p, m, force_change_index)
            })
    }
    fn select(&mut self, x_data: &Vec<f64>, y_data: &Vec<f64>) {
        let crossover_res_list = &self.crossover_res_list;
        self.particles
            .par_iter_mut()
            .zip(crossover_res_list)
            .for_each(|(p, c)| {
                p.select(x_data, y_data, c);
            });
    }
    pub fn evolution(
        &mut self,
        differential_weight: f64,
        crossover_probability: f64,
        x_data: &Vec<f64>,
        y_data: &Vec<f64>,
    ) {
        self.mutation(differential_weight);
        self.croseeover(crossover_probability);
        self.select(x_data, y_data);
    }
    pub fn result(&self, x_data: &Vec<f64>, y_data: &Vec<f64>)->(f64,Vec<f64>) {
        let mut best_loss: Option<f64> = None;
        let mut best_position_index: Option<usize> = None;
        self.particles
            .iter()
            .map(|x| evaluate(&x.position, x_data, y_data))
            .enumerate()
            .for_each(|(i, x)| match best_loss {
                None => {
                    best_loss = Some(x);
                    best_position_index = Some(i);
                }
                _ => match x <= best_loss.unwrap() {
                    true => {
                        best_loss = Some(x);
                        best_position_index = Some(i);
                    }
                    _ => (),
                },
            });
        (best_loss.unwrap(),
            self.particles
                .get(best_position_index.unwrap())
                .unwrap()
                .position.clone())

    }
}
