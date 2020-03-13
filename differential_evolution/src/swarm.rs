use crate::crossover_res::CrossoverResult;
use crate::evaluate;
use crate::mutation_force::MutationForce;
use crate::particle::Particle;
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
        let particles = &self.particles;
        let mutation_force_list = &self.mutation_force_list;
        let mut rng = thread_rng();
        let nums: Vec<usize> = (0..self.number_of_particles).collect();
        let force_change_index: usize = *nums.choose(&mut rng).unwrap();
        self.crossover_res_list
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, u)| {
                u.calc_crossover_res(
                    crossover_probability,
                    &particles[i],
                    &mutation_force_list[i],
                    force_change_index,
                )
            })
    }
    fn select(&mut self, x_data: &Vec<f64>, y_data: &Vec<f64>) {
        let crossover_res_list = &self.crossover_res_list;
        self.particles
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, p)| {
                p.select(x_data, y_data, &crossover_res_list[i]);
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
    pub fn result(&self, x_data: &Vec<f64>, y_data: &Vec<f64>) {
        println!(
            "{:?}",
            self.particles
                .iter()
                .map(|x| x.position[0])
                .collect::<Vec<f64>>()
        );
        let res = self.particles
                .iter()
                .map(|x| evaluate(&x.position, x_data, y_data))
                .collect::<Vec<f64>>();
        let mut best:Option<f64> = None;
        res.iter().for_each(|x|match best{
            None => best=Some(*x),
            _ => match *x<=best.unwrap() {
                true=> best = Some(*x),
                _=> ()
            }
        });
        
        println!(
            "{:?}",
            best.unwrap()
        );

    }
}
