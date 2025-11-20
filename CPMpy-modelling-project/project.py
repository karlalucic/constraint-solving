import pickle
from datetime import datetime

import cpmpy as cp
import networkx as nx
import numpy as np


def draw_tree(files, arc, labels=None, node_color=None, **kwargs):
    """Draw `files` nodes with the `arc` variable values. You can draw the tree with labels/colors."""

    G = nx.DiGraph()
    for i in range(files):
        G.add_node(i)
    for i in range(files):
        for j in range(files):
            if arc[i, j].value():
                G.add_edge(i, j)

    size = 250
    nx.draw(
        G,
        with_labels=True,
        node_size=[len(labels[v]) * size for v in G.nodes()],
        labels=labels,
        node_color=node_color,
        **kwargs,
    )

    return G


class UnexpectedError(Exception):
    pass


class ModelException(Exception):
    pass


def show_assignment(a):
    if a is False:
        return "UNSAT"
    return "\n".join(f"{x} = {a}" for x, a in a)


def solveAll(
    X, P, projected_solution_limit=None, time_limit=None, verbosity=1, diverse=False
):
    """Return all solutions of `P` as values of the projected variables, `X`, within the time limit, or `False` iff `P` is determined to be unsatisfiable."""
    sols = set()

    def value(x):
        def get_normalized_value(x):
            v = x.value()
            if v is None:
                return v
                # return x.lb  # TODO coerce correct type ..
            elif cp.expressions.utils.is_boolexpr(v):
                return bool(v)
            elif cp.expressions.utils.is_int(v):
                return int(v)
            else:
                raise TypeError("Unkown value type:", x, v, type(v))

        if isinstance(x, np.ndarray):
            return tuple(get_normalized_value(x_i) for x_i in x.flat)
        else:
            return get_normalized_value(x)

    class SolutionLimitReached(Exception):
        pass

    def store_sol():
        if all(x.value() is None for x in X):
            return

        sol = tuple((str(x.name), value(x)) for x in X if x.value() is not None)

        if sol not in sols:
            sols.add(sol)
            if verbosity >= 2:
                print(len(sols), end=" ", flush=True)

        if (
            projected_solution_limit is not None
            and len(sols) == projected_solution_limit
        ):
            if verbosity >= 2:
                print("\n", flush=True)
            raise SolutionLimitReached

    try:
        if verbosity >= 2:
            print("Solving your model, found solution: ", end="")
        if P.has_objective():
            raise NotImplementedError
            if P.solve(time_limit=time_limit):
                store_sol()
        else:
            if diverse:
                t = datetime.now()
                while True:
                    try:
                        P.solve(time_limit=time_limit)
                    except Exception as e:
                        raise ModelException(
                            "FAIL: your model raised an exception during solving."
                        ) from e

                    match P.status().exitstatus:
                        case cp.solvers.solver_interface.ExitStatus.UNSATISFIABLE:
                            if len(sols):
                                break  # all sols enumerated
                            else:
                                return False  # unsat
                        case cp.solvers.solver_interface.ExitStatus.UNKNOWN:
                            break
                        case (
                            cp.solvers.solver_interface.ExitStatus.FEASIBLE
                            | cp.solvers.solver_interface.ExitStatus.OPTIMAL
                        ):
                            store_sol()
                            if len(sols) == projected_solution_limit:
                                raise SolutionLimitReached

                            P += cp.any(x != x.value() for x in X)  # ensure termination
                            # euclidian diversity metric
                            P.maximize(
                                cp.sum(
                                    cp.sum(cp.abs(x - x.value()) for x in X)
                                    for sol in sols
                                )
                            )
                            if time_limit is not None:
                                nt = datetime.now()
                                time_limit -= (nt - t).total_seconds()
                                t = nt
                                if time_limit < 0:
                                    if verbosity >= 2:
                                        print("Solving time limit reached.")
                                    break
                        case _:
                            raise Exception("asdf")
            else:
                raise NotImplementedError
                P.solveAll(display=store_sol, time_limit=time_limit)
    except SolutionLimitReached:
        pass

    if verbosity >= 2:
        print("")
    return frozenset(sols)


def check(
    P,
    decisions,
    Qs,
    base=None,
    solution_limit=None,
    time_limit=None,
    verbosity=1,
    stages=None,
):
    try:
        n_stages = len(Qs)
        if stages is None:
            stages = range(n_stages)
        elif isinstance(stages, int):
            stages = range(stages)

        Qs = {stage: Qs[stage] for stage in sorted(stages)}

        if base is None:
            base = True

        # Copy to not alter student model
        P = P.copy()

        # Match variables from student model to known decision variables
        P_X = cp.transformations.get_variables.get_variables_model(P)
        X = {
            str(decision.name): next(
                (x for x in P_X if x.name == decision.name), decision
            )
            for decision in decisions
        }

        # ensure all decision variables are assigned and projected
        for x in X.values():
            P += [x == x]

        # add various objectives as heuristics to find violating solutions
        P_div = P.copy()
        if P_div.has_objective():
            P_div.objective_ = None

        assert not P_div.has_objective()

        # generate diverse set of solutions
        assignments = solveAll(
            [x for x in X.values() if x is not None],
            P_div,
            projected_solution_limit=solution_limit,
            time_limit=time_limit,
            verbosity=verbosity,
            diverse=True,
        )

        def enforce_assignment(x, v):
            """Enforce the assignment `x==v`."""
            if isinstance(x, np.ndarray):
                return cp.all(
                    x_i == v_i for x_i, v_i in zip(x.flat, v) if v_i is not None
                )
            else:
                if v is not None:
                    return x == v
            return True

        # stage to violation: either None (if passing), False (if student model is unsat), or return a MUS of the violation
        results = {}

        for stage, Q in Qs.items():
            if verbosity >= 3:
                print(
                    f"Checking stage {stage + 1}: ",
                    end="",
                )

            if assignments is False:
                print(f"- STAGE{stage + 1}: FAIL (Model unsatisfiable)")
                results[stage] = False
                continue

            violation = None

            # Check all found assignments
            if verbosity >= 2:
                print(f"Checking stage {stage + 1}, assignment:", end=" ")

            for i, assignment in enumerate(sorted(assignments), start=1):
                if verbosity >= 2:
                    print(i, end=" ")
                m = cp.Model(
                    Q,
                    base,
                    cp.all([enforce_assignment(X[x], a) for x, a in assignment]),
                )

                m.solve()
                match m.status().exitstatus:
                    case cp.solvers.solver_interface.ExitStatus.UNSATISFIABLE:
                        violation = assignment
                        print(
                            f"- STAGE{stage + 1}: FAIL (Submission allows invalid assignment)"
                        )

                        # Shouldn't we MUS it?
                        from cpmpy.tools.explain import mus

                        lits = []
                        lmap = {}  # reverse map literal to assignment
                        for x, a in assignment:
                            lit = enforce_assignment(X[x], a)
                            lits.append(lit)
                            lmap[lit] = (x, a)
                        sublits = mus(lits, hard=[Q, base])
                        subassgn = [lmap[lit] for lit in sublits]

                        print("  Invalid part of the assignment:")
                        print(f"{show_assignment(subassgn)}")

                        if verbosity >= 3:
                            print(f"Full violation: {show_assignment(violation)}")

                        results[stage] = subassgn
                        break
                    case cp.solvers.solver_interface.ExitStatus.UNKNOWN:
                        print(
                            f"- STAGE{stage + 1}: FAIL (Timeout (or other unknown exit statement))"
                        )
                        raise TimeoutError
                    case (
                        cp.solvers.solver_interface.ExitStatus.FEASIBLE
                        | cp.solvers.solver_interface.ExitStatus.OPTIMAL
                    ):
                        pass
                    case status:
                        print(
                            f"- STAGE{stage + 1}: FAIL (Unexpected status during checking: {status})"
                        )
                        raise Exception(f"Unexpected status during checking: {status}")

            if violation is None:
                print(f"- STAGE{stage + 1}: PASS")
                results[stage] = None

            if verbosity >= 3:
                print("")

        # clear variable values
        for x in P_X:
            x.clear()

        print("=========")
        grade = sum(1 for violation in results.values() if violation is None)
        print(f"Stages passed: {grade}/{n_stages}")
        print()

        return results
        # don't return anything, to not confuse notebook output
        # FIXME enable
        return

    except ModelException as e:
        raise e
    except Exception as e:
        if verbosity >= 3:
            raise e
        raise UnexpectedError(
            "ERROR: An unexpected error occured, please contact the teaching team with this notebook and stacktrace."
        ) from e


def check_fn(path, dummy=False, return_violations=False):
    def check_(
        P,
        solution_limit=100,
        time_limit=None,
        verbosity=1,
        stages=None,
    ):
        if dummy:
            return
        _, minor, patch = (int(i) for i in cp.__version__.split("."))
        assert minor >= 9 and patch >= 28, (
            f"The project requires version `cpmpy>=0.9.28`, but version `cpmpy=={cp.__version__}` is currently installed. Please `pip install --upgrade cpmpy` to get the latest version."
        )
        with open(path, "rb") as file:
            decisions, Qs, base = pickle.load(file)

        violations = check(
            P,
            decisions,
            Qs,
            base=base,
            solution_limit=solution_limit,
            time_limit=time_limit,
            verbosity=verbosity,
            stages=stages,
        )
        if return_violations:
            return violations

    return check_


check1 = check_fn("project1_1.pkl")
check2 = check_fn("project1_2.pkl")
