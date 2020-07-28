use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rand::Rng;
use rayon::prelude::*;
/// 这是粒子群算法中的粒子
#[derive(Debug)]
pub struct Particle {
    /// 粒子的位置
    pub position: Vec<f64>,
    /// 位置的上边界
    position_max: Vec<f64>,
    /// 位置的下边界
    position_min: Vec<f64>,
    /// 粒子的速度
    pub velocity: Vec<f64>,
    /// 速度的上限
    velocity_max: Vec<f64>,
    /// 速度的反方向上限
    velocity_min: Vec<f64>,
    /// 历史中的最好位置
    pub local_best_position: Vec<f64>,
    /// 历史中的最好偏差
    pub local_best_loss: f64,
}
impl Particle {
    /// 构建一个新粒子
    pub fn new(position_max: &[f64], position_min: &[f64], x_data: &[f64], y_data: &[f64]) -> Self {
        // 将位置上下限拷贝
        let p_max = position_max.to_owned();
        let p_min = position_min.to_owned();
        // 计算出位置上下限之间的距离
        let position_range: Vec<f64> = p_max.iter().zip(&p_min).map(|(a, b)| a - b).collect();
        // 以均匀分布选取粒子的初始位置
        let mut rng = rand::thread_rng();
        let position: Vec<f64> = position_range
            .iter()
            .map(|a| a * rng.gen::<f64>())
            .zip(&p_min)
            .map(|(a, b)| a + b)
            .collect();
        // 粒子的初始速度设为位置极差的-0.1倍
        let velocity: Vec<f64> = position_range.iter().map(|p| -0.1 * p).collect();
        // 速度上限（正反方向）设为位置极差的1/5
        let velocity_max = position_range.iter().map(|a| a / 5.0).collect();
        let velocity_min = position_range.iter().map(|a| -a / 5.0).collect();
        // 初始的历史最佳位置就是当前位置
        let local_best_position = position.clone();
        // 初始的历史最佳偏差由当前位置和待拟合数据算出
        let local_best_loss = calc_loss(&position, &x_data, &y_data);
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
    /// 粒子的运动
    pub fn evolution(
        &mut self,
        w: f64,
        c1: f64,
        c2: f64,
        global_best_position: &[f64],
        x_data: &[f64],
        y_data: &[f64],
    ) {
        let mut rng = rand::thread_rng();
        // 更新速度 v = w * v + c1 * rand * (lbp - p) + c2 * rand * (gbp - p)。控制速度的上限。
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
            .map(|((v, vmax), vmin)| {
                if v > *vmax {
                    *vmax
                } else if v < *vmin {
                    *vmin
                } else {
                    v
                }
            })
            .collect();
        // 更新位置 p = p + v。控制位置的上下限。 
        self.position = self
            .position
            .iter()
            .zip(&self.velocity)
            .map(|(p, v)| p + v)
            .zip(&self.position_max)
            .zip(&self.position_min)
            .map(|((p, pmax), pmin)| {
                if p > *pmax {
                    *pmax
                } else if p < *pmin {
                    *pmin
                } else {
                    p
                }
            })
            .collect();
        // 计算新位置的偏差
        let loss = calc_loss(&self.position, &x_data, &y_data);
        // 如果新的偏差好于历史最佳，则将历史最佳偏差和位置更新
        if loss < self.local_best_loss {
            self.local_best_loss = loss;
            self.local_best_position = self.position.clone();
        }
    }
    /// 将粒子的位置加到一个数组上去
    fn add_position_to_vec(&self, sum_of_position: &mut [f64]) {
        sum_of_position
            .iter_mut()
            .zip(&self.position)
            .for_each(|(sop, p)| *sop += p)
    }
    /// 将粒子的位置距离平均位置的平方偏差加到一个数组上去
    fn add_position_square_error_to_vec(
        &self,
        sum_of_square_error: &mut [f64],
        mean_position: &[f64],
    ) {
        sum_of_square_error
            .iter_mut()
            .zip(&self.position)
            .zip(mean_position)
            .for_each(|((sose, p), mp)| *sose += (mp - p).powi(2))
    }
}
/// 一个粒子群
#[derive(Debug)]
pub struct Swarm {
    /// 粒子的总数
    pub num_of_particles: usize,
    /// 粒子的列表
    pub particle_list: Vec<Particle>,
    /// 全局历史最佳位置
    pub global_best_position: Vec<f64>,
    /// 全局历史最佳偏差
    pub global_best_loss: f64,
    /// 待拟合的数据
    x_data: Vec<f64>,
    y_data: Vec<f64>,
}

impl Swarm {
    /// 创建一个新粒子群
    pub fn new(
        num_of_particles: usize,
        position_max: &[f64],
        position_min: &[f64],
        x_data: &[f64],
        y_data: &[f64],
    ) -> Self {
        // 构造一个向量把粒子创建好然后推进去
        let mut particle_list: Vec<Particle> = Vec::new();
        particle_list.reserve(num_of_particles);
        for _ in 0..num_of_particles {
            particle_list.push(Particle::new(position_max, position_min, x_data, y_data));
        }
        // 比较出当前（所有粒子均未运动）的历史最佳位置和偏差
        let (global_best_loss, global_best_position) = compare(&particle_list, None);
        Swarm {
            num_of_particles,
            particle_list,
            global_best_loss,
            global_best_position,
            x_data: x_data.to_vec(),
            y_data: y_data.to_vec(),
        }
    }
    /// 粒子群的演化
    pub fn evolution(&mut self, w: f64, c1: f64, c2: f64) {
        let gbp = &self.global_best_position;
        let x_data = &self.x_data;
        let y_data = &self.y_data;
        // 每一个粒子各自运动
        self.particle_list
            // .iter_mut()
            .par_iter_mut()
            .for_each(|p| p.evolution(w, c1, c2, gbp, x_data, y_data));
        // 运动完之后更新历史最佳位置和偏差
        let (global_best_loss, global_best_position) =
            compare(&self.particle_list, Some(self.global_best_loss));
        self.global_best_position = global_best_position;
        self.global_best_loss = global_best_loss;
    }
    /// 测量粒子群的位置的平均值和标准差
    pub fn result_evaluate(&self) -> (Vec<f64>, Vec<f64>) {
        // 测量平均值
        let position_len = self.global_best_position.len();
        let mut sum_of_position = vec![0.0; position_len];
        self.particle_list
            .iter()
            .for_each(|ptcl| ptcl.add_position_to_vec(&mut sum_of_position));
        let mean_position = sum_of_position
            .iter()
            .map(|sop| *sop / self.num_of_particles as f64)
            .collect::<Vec<f64>>();
        // 测量标准差
        let mut sum_of_square_error = vec![0.0; position_len];
        self.particle_list.iter().for_each(|ptcl| {
            ptcl.add_position_square_error_to_vec(&mut sum_of_square_error, &mean_position)
        });
        let standard_var_of_position = sum_of_square_error
            .iter()
            .map(|sose| (*sose / self.num_of_particles as f64).sqrt())
            .collect::<Vec<f64>>();
        (mean_position, standard_var_of_position)
    }
}
/// 比较粒子群和历史最佳偏差，返回新的历史最佳偏差和历史最佳位置
pub fn compare(
    particle_list: &[Particle],
    current_global_best_loss: Option<f64>,
) -> (f64, Vec<f64>) {
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
        candidate_global_best_position.unwrap(),
    )
}
/// 由粒子位置和待拟合数据算出平方偏差
pub fn calc_loss(particle_position: &[f64], x_data: &[f64], y_data: &[f64]) -> f64 {
    // 取出参数
    let a = particle_position[0];
    let b = particle_position[1];
    // x带入函数得到预测y，计算预测y和实际y的平方偏差，最后每个数据点的偏差相加
    x_data
        .iter()
        .map(|x| a/(b+x.powi(2)))
        .zip(y_data)
        .map(|(y_pred, y_data)| (y_pred - y_data).powi(2))
        .sum()
}
/// 导出的粒子群优化函数
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
) -> PyResult<(f64, Vec<f64>, Vec<f64>, Vec<f64>)> {
    // 创建粒子群
    let mut a_swarm = Swarm::new(
        num_of_particles,
        &position_max,
        &position_min,
        &x_data,
        &y_data,
    );
    // 粒子群演化
    for _ in 0..steps {
        a_swarm.evolution(w, c1, c2);
    }
    // 测量粒子群的位置的平均值和标准差
    let (mean_position, standard_var_of_position) = a_swarm.result_evaluate();
    Ok((
        a_swarm.global_best_loss,
        a_swarm.global_best_position,
        mean_position,
        standard_var_of_position,
    ))
}

#[pymodule]
fn lorentzian(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(classic_pso))?;

    Ok(())
}
