use std::{fmt::Display, fs::read_to_string, path::Path};

use mdp_brkga::{CurrentDecoder, ExperimentalDecoder, MaximumDiversity};
use ndarray::Array2;
use optimum::{
    core::Problem,
    metaheuristics::genetic::{Brkga, BrkgaParams, Decoder},
};
use rand::SeedableRng;

fn main() -> std::io::Result<()> {
    //let problem = load_input(Path::new("MDG-b_21_n2000_m200.txt"))?;
    let problem = load_input(Path::new("matrizn500_m200.txt"))?;

    if std::env::args().count() == 2 {
        println!("Using new decoder");
        let decoder = ExperimentalDecoder::new(&problem);
        run(decoder, problem.solution_size);
    } else {
        println!("Using current decoder");
        let decoder = CurrentDecoder::new(&problem);
        run(decoder, problem.input_size);
    }

    Ok(())
}

fn run<D: Decoder>(decoder: D, members_size: usize)
where
    <D::P as Problem>::Value: Display,
{
    let rng = rand_pcg::Pcg64::seed_from_u64(1);

    let params = BrkgaParams {
        population_size: 100.try_into().unwrap(),
        members: members_size.try_into().unwrap(),
        elites: 20,
        mutants: 30,
        crossover_bias: 0.8,
    };
    let mut brkga = Brkga::new(&decoder, rng, params);

    println!("Initial best: {}", brkga.best().value);

    let total_generations = 1000;
    for _ in 0..total_generations {
        brkga.evolve();
    }

    println!("Final best: {}", brkga.best().value);
}

fn load_input(path: &Path) -> std::io::Result<MaximumDiversity> {
    let file_name = { path.file_name().unwrap().to_string_lossy() };

    let (_, last) = file_name.split_once('n').unwrap();
    let (input_size, last) = last.split_once('_').unwrap();

    let input_size: usize = input_size.parse().unwrap();
    let solution_size: usize = last[1..].strip_suffix(".txt").unwrap().parse().unwrap();

    let input = read_to_string(path)?;

    let mut lines = input.lines();
    // skip first line, which is the problem size.
    lines.next();

    let mut vec = Vec::<f64>::with_capacity(input_size * input_size);

    for line in lines {
        let elements = line
            .split_ascii_whitespace()
            .map(|v| v.parse::<f64>().unwrap());

        vec.extend(elements);
    }

    Ok(MaximumDiversity {
        matrix: Array2::from_shape_vec((input_size, input_size), vec).unwrap(),
        solution_size,
        input_size,
    })
}
