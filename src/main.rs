mod cli;

use std::{fs::read_to_string, path::Path, time::Duration};

use clap::StructOpt;
use cli::{DecoderChooser, Opts};
use mdp_brkga::{CurrentDecoder, ExperimentalDecoder, MaximumDiversity};
use ndarray::Array2;
use optimum::{
    analysis::batch::{Batch, Statistics},
    core::solver::hook,
    core::stop_criterion::TimeCriterion,
    metaheuristics::genetic::{
        brkga::{Brkga, Params},
        Decoder,
    },
};
use rand::SeedableRng;

fn main() -> std::io::Result<()> {
    let opts = Opts::parse();

    let problem = load_input(&opts.problem)?;

    let create_params = |member_size: usize| Params {
        population_size: opts.population_size.try_into().unwrap(),
        member_size: member_size.try_into().unwrap(),
        elites: opts.elites,
        mutants: opts.mutants,
        crossover_bias: opts.crossover_bias,
    };

    match opts.decoder {
        DecoderChooser::New => {
            println!("Using new decoder");
            let decoder = ExperimentalDecoder::new(&problem);
            run(decoder, create_params(problem.solution_size), opts.seed);
        }
        DecoderChooser::Current => {
            println!("Using current decoder");
            let decoder = CurrentDecoder::new(&problem);
            run(decoder, create_params(problem.input_size), opts.seed);
        }
    }

    Ok(())
}

fn run<D: Decoder<P = MaximumDiversity>>(decoder: D, params: Params, seed: usize) {
    let stop_criterion = TimeCriterion::new(Duration::from_secs(1));

    let build_solver = |seed, exec_number| {
        let rng = rand_pcg::Pcg64::seed_from_u64((seed + exec_number) as u64);

        Brkga::new(&decoder, rng, params)
    };

    let batch = Batch::new(seed, 10, build_solver, &stop_criterion, hook::Empty).unwrap();

    let statistics = Statistics::new(&batch);
    let (_, best, _) = statistics.best();
    println!(
        "Final best: {}, average value: {}, average time: {}",
        best.value(),
        statistics.average_value(),
        statistics.average_time().as_secs_f64(),
    );
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
