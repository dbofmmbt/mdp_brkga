use std::cell::RefCell;

use optimum::{
    core::Problem,
    metaheuristics::genetic::{Decoder, Key},
};
use ordered_float::NotNan;

use crate::{MaximumDiversity, MdpSolution};

pub struct CurrentDecoder<'a> {
    auxiliary: RefCell<Vec<usize>>,
    problem: &'a MaximumDiversity,
}

impl<'a> CurrentDecoder<'a> {
    pub fn new(problem: &'a MaximumDiversity) -> Self {
        Self {
            auxiliary: RefCell::new(vec![0; problem.input_size]),
            problem,
        }
    }
}

impl Decoder for CurrentDecoder<'_> {
    type P = MaximumDiversity;

    fn decode(&self, member: &[Key]) -> <Self::P as optimum::core::Problem>::Value {
        let mut aux = self.auxiliary.borrow_mut();

        aux.iter_mut().enumerate().for_each(|(i, el)| {
            *el = i;
        });

        aux.sort_unstable_by_key(|&element| unsafe { NotNan::new_unchecked(member[element]) });

        let solution = MdpSolution {
            elements: aux[0..self.problem.solution_size].to_owned(),
        };

        self.problem.objective_function(&solution)
    }
}
