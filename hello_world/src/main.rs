use rand::Rng;
#[derive(Debug)]
struct Bound {
    min: Vec<f64>,
    max: Vec<f64>,
}
impl Bound {
    fn new_bound() -> Bound {
        Bound {
            min: vec![-1.0, -1.0],
            max: vec![2.0, 1.0],
        }
    }

    fn bound_len(&self) -> usize {
        self.min.len()
    }

    fn range(&self) -> Vec<f64> {
        let mut bound_range: Vec<f64> = vec![0.0; self.bound_len()];
        for i in 0..self.bound_len() {
            bound_range[i] = self.max[i] - self.min[i];
        }
        return bound_range;
    }
}

#[derive(Debug)]
struct Particle {
    position: Vec<f64>,
    velocity: Vec<f64>,
    local_best_position: Vec<f64>,
    local_best_loss: f64,
    position_bound: Bound,
    velocity_bound: Bound,
}
impl Particle {
    fn new_particle() -> Particle {
        let position_bound = Bound::new_bound();
        let position_bound_len = position_bound.bound_len();
        let position_range = position_bound.range();
        let mut velocity_bound = Bound::new_bound();
        let mut position: Vec<f64> = vec![0.0; position_bound_len];
        let mut velocity: Vec<f64> = vec![0.0; position_bound_len];
        let mut rng = rand::thread_rng();
        for i in 0..position_bound_len {
            position[i] = rng.gen::<f64>() * position_range[i] + position_bound.min[i];
            velocity[i] = -1.0 * position[i] / 10.0;
            velocity_bound.max[i] = velocity_bound.max[i] / 10.0;
            velocity_bound.min[i] = velocity_bound.max[i] * -1.0;
        }
        let local_best_position = position.clone();
        let local_best_loss = position[0].powi(2) + position[1].powi(2);
        Particle {
            position,
            velocity,
            local_best_position,
            local_best_loss,
            position_bound,
            velocity_bound,
        }
    }
    fn evaluate(&self) -> f64 {
        self.position[0].powi(2) + self.position[1].powi(2)
    }
    fn evolution(&mut self, w: f64, c1: f64, c2: f64, global_best_position: &Vec<f64>) {
        let mut rng = rand::thread_rng();
        let position_len = self.position.len();
        for i in 0..position_len {
            self.velocity[i] = w * self.position[i]
                + c1 * rng.gen::<f64>() * (self.local_best_position[i] - self.position[i])
                + c2 * rng.gen::<f64>() * (global_best_position[i] - self.position[i]);
            if self.velocity[i] > self.velocity_bound.max[i] {
                self.velocity[i] = self.velocity_bound.max[i];
            }
            if self.velocity[i] < self.velocity_bound.min[i] {
                self.velocity[i] = self.velocity_bound.min[i];
            }
        }
        for i in 0..position_len {
            self.position[i] = self.position[i] + self.velocity[i];
            if self.position[i] > self.position_bound.max[i] {
                self.position[i] = self.position_bound.max[i];
            }
            if self.position[i] < self.position_bound.min[i] {
                self.position[i] = self.position_bound.min[i];
            }
        }
        let loss = self.evaluate();
        if loss <= self.local_best_loss {
            for i in 0..position_len {
                self.local_best_position[i] = self.position[i];
            }
            self.local_best_loss = loss;
        }
    }
}
#[derive(Debug)]
struct Swarm {
    particles: Vec<Particle>,
    global_best_position: Vec<f64>,
    global_best_loss: f64,
}
impl Swarm {
    fn new_swarm(num_of_particles: usize) -> Swarm {
        let a_particle = Particle::new_particle();
        let position_len = a_particle.position.len();
        let mut global_best_position: Vec<f64> = vec![0.0; position_len];
        let mut global_best_loss = 0.0;
        let mut particles: Vec<Particle> = Vec::new();
        particles.reserve(num_of_particles);
        for i in 0..num_of_particles {
            let a_particle = Particle::new_particle();
            if i == 0 {
                global_best_position = a_particle.local_best_position.clone();
                global_best_loss = a_particle.local_best_loss.clone();
            } else {
                if a_particle.local_best_loss <= global_best_loss {
                    global_best_position = a_particle.local_best_position.clone();
                    global_best_loss = a_particle.local_best_loss.clone();
                }
            }
            particles.push(a_particle);
        }
        Swarm {
            particles,
            global_best_position,
            global_best_loss,
        }
    }
    fn evolution(&mut self, w: f64, c1: f64, c2: f64) {
        let num_of_particles = self.particles.len();
        for i in 0..num_of_particles {
            self.particles[i].evolution(w, c1, c2, &self.global_best_position);
        }
        for i in 0..num_of_particles {
            if self.particles[i].local_best_loss <= self.global_best_loss {
                self.global_best_loss = self.particles[i].local_best_loss.clone();
                self.global_best_position = self.particles[i].local_best_position.clone();
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 1 {
        println!("pso num_of_particles num_of_step w c1 c2" );
    }else{
        let num_of_particles:usize = args[1].parse().unwrap();
        let num_of_step:usize = args[2].parse().unwrap();
        let w:f64 = args[3].parse().unwrap();
        let c1:f64 = args[4].parse().unwrap();
        let c2:f64 = args[5].parse().unwrap();
        let mut a_swarm = Swarm::new_swarm(num_of_particles);
        for i in 0..num_of_step{
            a_swarm.evolution(w, c1,c2);
            println!("{:?}",&a_swarm.global_best_position);
        }
    }
}
