import logging
import ConfigSpace as CS
import numpy as np
import pandas as pd

import mlos_core.optimizers
from axe.lsm.cost import EndureCost
from axe.lcm.data.generator import KHybridGenerator
from axe.lsm.solver import KLSMSolver
from axe.lsm.types import LSMDesign, System, Workload, LSMBounds
from axe.data.io import Reader


def export_to_csv(mlos_costs, analytical_costs, mlos_designs, analytical_designs, systems, workloads) -> None:
    data = {
        "MLOS_Costs": mlos_costs,
        "Analytical_Costs": analytical_costs,
        **{f"MLOS_Design_{attr}": [getattr(d, attr) for d in mlos_designs] for attr in vars(LSMDesign())},
        **{f"Analytical_Design_{attr}": [getattr(d, attr) for d in analytical_designs] for attr in vars(LSMDesign())},
        **{f"System_{attr}": [getattr(s, attr) for s in systems] for attr in vars(System())},
        **{f"Workload_{attr}": [getattr(w, attr) for w in workloads] for attr in vars(Workload())}
    }
    df = pd.DataFrame(data)
    df.to_csv("output_results.csv", index=False)


class BayesianPipelineMlos:
    def __init__(self, conf: dict) -> None:
        self.end_time: float = 0
        self.conf = conf
        self.start_time: float = 0
        self.run_id: int = 0
        self.conn = None
        self.log: logging.Logger = logging.getLogger(conf["log"]["name"])
        lsm_bounds_config = conf["lsm"]["bounds"]
        self.bounds: LSMBounds = LSMBounds(
            max_considered_levels=lsm_bounds_config["max_considered_levels"],
            bits_per_elem_range=(lsm_bounds_config["memory_budget_range"][0], lsm_bounds_config["memory_budget_range"][1]),  # if you decide to use it here
            size_ratio_range=(lsm_bounds_config["size_ratio_range"][0], lsm_bounds_config["size_ratio_range"][1]),
            page_sizes=tuple(lsm_bounds_config["page_sizes"]),
            entry_sizes=tuple(lsm_bounds_config["entry_sizes"]),
            memory_budget_range=(lsm_bounds_config["memory_budget_range"][0], lsm_bounds_config["memory_budget_range"][1]),
            selectivity_range=(lsm_bounds_config["selectivity_range"][0], lsm_bounds_config["selectivity_range"][1]),
            elements_range=(lsm_bounds_config["elements_range"][0], lsm_bounds_config["elements_range"][1])
        )
        self.cf: EndureCost = EndureCost(self.bounds.max_considered_levels)
        self.num_k_values = self.conf["job"]["BayesianOptimization"]["num_k_values"]
        self.generator = KHybridGenerator(self.bounds)
        self.optimizer = self.conf["job"]["BayesianOptimization"]["mlos"]["optimizer"]
        self.n_runs = self.conf["job"]["BayesianOptimization"]["mlos"]["num_runs"]
        self.num_iterations = self.conf["job"]["BayesianOptimization"]["mlos"]["iteration"]

    def run(self,):
        mlos_costs = []
        mlos_designs = []
        analytical_costs = []
        analytical_designs = []
        systems = []
        workloads = []
        input_space = define_config_space(self.num_k_values, self.bounds)
        optimizer = self.select_optimizer(self.optimizer, input_space)
        for i in range(self.n_runs):
            print(f"Iteration {i + 1}/{self.n_runs} running")
            system = self.generator._sample_system()
            systems.append(system)
            z0, z1, q, w = self.generator._sample_workload(4)
            workload = Workload(z0=z0, z1=z1, q=q, w=w)
            workloads.append(workload)
            best_observation = self.run_optimization_loop(self.num_iterations, optimizer, system, workload)
            best_design = self.interpret_optimizer_result(best_observation)
            mlos_cost = self.target_function(best_design, system, workload)
            mlos_costs.append(mlos_cost)
            mlos_designs.append(best_design)
            design_analytical, cost_analytical = find_analytical_results(system, workload, self.bounds)
            analytical_costs.append(cost_analytical)
            analytical_designs.append(design_analytical)
        export_to_csv(mlos_costs, analytical_costs, mlos_designs, analytical_designs, systems, workloads)
        return mlos_costs, analytical_costs, mlos_designs, analytical_designs, systems, workloads

    def interpret_optimizer_result(self, observation) -> LSMDesign:
        h = observation['h'].iloc[0] if isinstance(observation['h'], pd.Series) else observation['h']
        T = observation['t'].iloc[0] if isinstance(observation['t'], pd.Series) else observation['t']
        k_values = [observation[f'k_{i}'].iloc[0] if isinstance(observation[f'k_{i}'], pd.Series)
                    else observation[f'k_{i}'] for i in range(self.num_k_values)]
        design = LSMDesign(h=h, T=T, K=k_values)
        return design

    def select_optimizer(self, optimizer: str, input_space):
        if optimizer == "Smac":
            return mlos_core.optimizers.SmacOptimizer(parameter_space=input_space)
        elif optimizer == "Flaml":
            return mlos_core.optimizers.FlamlOptimizer(parameter_space=input_space)
        elif optimizer == "Random":
            return mlos_core.optimizers.RandomOptimizer(parameter_space=input_space)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer}")

    def target_function(self, design: LSMDesign, system: System, workload: Workload) -> float:
        return self.cf.calc_cost(design, system, workload.z0, workload.z1, workload.q, workload.w)

    def run_optimization_loop(self, iteration, optimizer, system: System, workload: Workload):
        for _ in range(iteration):
            optimizer = self.run_optimization(optimizer, system, workload)
        best_observation = optimizer.get_best_observation()
        return best_observation

    def run_optimization(self, optimizer, system: System, workload: Workload):
        suggested_config = optimizer.suggest()
        design = self.interpret_optimizer_result(suggested_config)
        cost = self.target_function(design, system, workload)
        config_data = {
            'h': [design.h],
            't': [design.T]
        }
        config_data.update({f'k_{i}': [design.K[i]] for i in range(len(design.K))})
        config_df = pd.DataFrame(config_data)
        score_series = pd.Series([cost])
        optimizer.register(config_df, score_series)
        return optimizer


def define_config_space(num_k_values: int, bounds: LSMBounds) -> CS.ConfigurationSpace:
    input_space = CS.ConfigurationSpace(seed=1234)
    # input_space.add_hyperparameter(CS.CategoricalHyperparameter
    # ("policy", ["Tiering", "Leveling", "Classic", "KHybrid", "QFixed", "YZHybrid"]))
    input_space.add_hyperparameter(CS.UniformFloatHyperparameter(name='h', lower=bounds.bits_per_elem_range[0],
                                                                 upper=bounds.bits_per_elem_range[1]))
    input_space.add_hyperparameter(CS.UniformIntegerHyperparameter(name='t', lower=bounds.size_ratio_range[0],
                                                                   upper=bounds.size_ratio_range[1]))
    for i in range(num_k_values):
        input_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                name=f'k_{i}',
                lower=1,
                upper=bounds.size_ratio_range[1] - 1
            )
        )
    return input_space


def find_analytical_results(system: System, workload: Workload, bounds: LSMBounds):
    solver = KLSMSolver(bounds)
    z0, z1, q, w = workload.z0, workload.z1, workload.q, workload.w
    nominal_design, nominal_solution = solver.get_nominal_design(system, z0, z1, q, w)
    k_values = nominal_design.K
    scalars = np.array([nominal_design.h, nominal_design.T])
    all_values = np.concatenate((scalars, k_values))
    cost_ana = solver.nominal_objective(all_values, system, z0, z1, q, w)
    print("Cost for the nominal design using analytical solver:", cost_ana)
    print("Nominal Design suggested by analytical solver:", nominal_design)
    return nominal_design, cost_ana


if __name__ == "__main__":
    config = Reader.read_config("axe.toml")

    log = logging.getLogger(config["log"]["name"])
    log.info("Initializing Bayesian Optimization Job")

    bayesian_optimizer = BayesianPipelineMlos(config)
    bayesian_optimizer.run()
