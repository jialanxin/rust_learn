use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rand::Rng;
use rayon::prelude::*;
#[derive(Debug)]
struct Particle {
    position: Vec<f64>,
    position_max: Vec<f64>,
    position_min: Vec<f64>,
    velocity: Vec<f64>,
    velocity_max: Vec<f64>,
    velocity_min: Vec<f64>,
    local_best_position: Vec<f64>,
    local_best_loss: f64,
}
impl Particle {
    fn new(position_max: &[f64], position_min: &[f64], x_data: &[f64], y_data: &[f64]) -> Self {
        let p_max = position_max.to_owned();
        let p_min = position_min.to_owned();
        let position_range: Vec<f64> = p_max.iter().zip(&p_min).map(|(a, b)| a - b).collect();
        let mut rng = rand::thread_rng();
        let position: Vec<f64> = position_range
            .iter()
            .map(|a| a * rng.gen::<f64>())
            .zip(&p_min)
            .map(|(a, b)| a + b)
            .collect();
        let velocity: Vec<f64> = position_range.iter().map(|p| -0.1 * p).collect();
        let velocity_max = position_range.iter().map(|a| a / 5.0).collect();
        let velocity_min = position_range.iter().map(|a| -a / 5.0).collect();
        let local_best_position = position.clone();
        let local_best_loss = evaluate(&position, &x_data, &y_data);
        Particle {
            position,
            position_max: p_max,
            position_min: p_min,
            velocity,
            velocity_max,
            velocity_min,
            local_best_position,
            local_best_loss,
        }
    }
    fn evolution(
        &mut self,
        w: f64,
        c1: f64,
        c2: f64,
        global_best_position: &[f64],
        x_data: &[f64],
        y_data: &[f64],
    ) {
        let mut rng = rand::thread_rng();
        self.velocity = self
            .position
            .iter()
            .zip(&self.local_best_position)
            .zip(global_best_position)
            .zip(&self.velocity)
            .map(|(((p, lbp), gbp), v)| {
                w * v + c1 * rng.gen::<f64>() * (lbp - p) + c2 * rng.gen::<f64>() * (gbp - p)
            })
            .zip(&self.velocity_max)
            .zip(&self.velocity_min)
            .map(|((v, &vmax), &vmin)| {
                if v > vmax {
                    vmax
                } else if v < vmin {
                    vmin
                } else {
                    v
                }
            })
            .collect();
        self.position = self
            .position
            .iter()
            .zip(&self.velocity)
            .map(|(p, v)| p + v)
            .zip(&self.position_max)
            .zip(&self.position_min)
            .map(|((p, &pmax), &pmin)| {
                if p > pmax {
                    pmax
                } else if p < pmin {
                    pmin
                } else {
                    p
                }
            })
            .collect();
        let loss = evaluate(&self.position, &x_data, &y_data);
        if loss < self.local_best_loss {
            self.local_best_loss = loss;
            self.local_best_position = self.position.clone();
        }
    }
}

#[derive(Debug)]
struct Swarm {
    num_of_particles: usize,
    particle_list: Vec<Particle>,
    global_best_position: Vec<f64>,
    global_best_loss: f64,
    x_data: Vec<f64>,
    y_data: Vec<f64>,
}

impl Swarm {
    fn new(
        num_of_particles: usize,
        position_max: &[f64],
        position_min: &[f64],
        x_data: &[f64],
        y_data: &[f64],
    ) -> Self {
        let mut particle_list: Vec<Particle> = Vec::new();
        particle_list.reserve(num_of_particles);
        for _ in 0..num_of_particles {
            particle_list.push(Particle::new(position_max, position_min, x_data, y_data));
        }
        let (global_best_loss, global_best_position) = compare(&particle_list, None);
        Swarm {
            num_of_particles,
            particle_list,
            global_best_loss,
            global_best_position: global_best_position.unwrap(),
            x_data: x_data.to_vec(),
            y_data: y_data.to_vec(),
        }
    }
    fn evolution(&mut self, w: f64, c1: f64, c2: f64) {
        let gbp = &self.global_best_position;
        let x_data = &self.x_data;
        let y_data = &self.y_data;
        self.particle_list
            // .iter_mut()
            .par_iter_mut()
            .for_each(|p| p.evolution(w, c1, c2, gbp, x_data, y_data));
        let (global_best_loss, global_best_position) =
            compare(&self.particle_list, Some(self.global_best_loss));
        self.global_best_position = global_best_position.unwrap();
        self.global_best_loss = global_best_loss;
    }
}
fn compare(
    particle_list: &[Particle],
    current_global_best_loss: Option<f64>,
) -> (f64, Option<Vec<f64>>) {
    let mut candidate_global_best_loss = current_global_best_loss;
    let mut candidate_global_best_position = None;
    particle_list
        .iter()
        .for_each(|a| match candidate_global_best_loss {
            None => {
                candidate_global_best_loss = Some(a.local_best_loss);
                candidate_global_best_position = Some(a.local_best_position.clone());
            }
            Some(x) => {
                if a.local_best_loss <= x {
                    candidate_global_best_loss = Some(a.local_best_loss);
                    candidate_global_best_position = Some(a.local_best_position.clone());
                }
            }
        });
    (
        candidate_global_best_loss.unwrap(),
        candidate_global_best_position,
    )
}
fn evaluate(particle_position: &[f64], x_data: &[f64], y_data: &[f64]) -> f64 {
    let a1 = particle_position[0];
    let a2 = particle_position[1];
    let e1 = particle_position[2];
    let e2 = particle_position[3];
    x_data
        .iter()
        .map(|x| (-x / e1).exp() * a1 + (-x / e2).exp() * a2)
        .zip(y_data)
        .map(|(y_pred, y_data)| (y_pred - y_data).powi(2))
        .sum()
}
#[pyfunction]
pub fn classic_pso(
    position_max: Vec<f64>,
    position_min: Vec<f64>,
    x_data: Vec<f64>,
    y_data: Vec<f64>,
    num_of_particles: usize,
    w: f64,
    c1: f64,
    c2: f64,
    steps: usize,
) -> PyResult<(f64, Vec<f64>)> {
    let mut a_swarm = Swarm::new(
        num_of_particles,
        &position_max,
        &position_min,
        &x_data,
        &y_data,
    );
    for _ in 0..steps {
        a_swarm.evolution(w, c1, c2);
    }
    Ok((a_swarm.global_best_loss, a_swarm.global_best_position))
}

#[pymodule]
fn pso(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(classic_pso))?;

    Ok(())
}
