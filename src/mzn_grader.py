import base64
import json
import logging
import os
import re
import sys
import zlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from json import JSONDecodeError
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

import minizinc

EMPTY_ERROR = (
    "The found solution appears to be empty.\n\nCheck your "
    "output statement and make sure it meets the requirements of "
    "the assignment. If the problem persists, then please ask "
    "your course instructor for help. "
)

GRADER_ERROR = (
    "An error occurred within the grader, please inform your course "
    "instructor.\n\nThe course instructor will need to for which "
    "assignment the error occurred, and at what time you made your "
    "submission. Thank you for your help and your patience. We hope to "
    "prevent these issues from happening in the future."
)

MODEL_ERROR = (
    "An error occurred while solving your "
    "model.\n\nPlease ensure that your MiniZinc model "
    "compiles correctly and works for all provided "
    "instances. If the problem persists, then please ask "
    "your course instructor for help."
)

SOLUTION_ERROR = (
    "An error occurred while solving your model.\n\nEnsure that your "
    "model does not contain any elements that are not supported by "
    "the solver and check that your model returns no error message "
    "when running locally. If the problem persists, then please ask "
    "your course instructor for help."
)

UNSAT_ERROR = (
    "Your model reported the problem as unsatisfiable, but the "
    "problem is satisfiable.\n\nPlease ensure that your model "
    "contains only the constraints that are part of the model "
    "description."
)

INPUT_ERROR = (
    "An error occurred while checking your solution.\n\nPlease "
    "ensure your assignment has been submitted correctly using the "
    "provided checker or output statement. If the problem persists, "
    "then please ask your course instructor for help."
)

UNKNOWN_MSG = (
    "Your submission is unable to find a feasible "
    "solution to the problem within the set time limit."
)

UNSAT_MSG = (
    "Congratulations! Your model correctly proved that the "
    "problem instance is unsatisfiable."
)

GRADER_LAPSE = (
    "The grader marked a solution for this instance as correct, but it was marked as "
    "unsatisfiable. "
)

GRADER_CHECKER_LAPSE = (
    "The grader appears to have a definition for a checkStatistics function, but no "
    "statisticsCheck field was found."
)


@dataclass
class Feedback:
    fractionalScore: float = 0.0
    feedback: str = GRADER_ERROR

    def serialise(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in {f.name for f in fields(cls)}}
        )


@dataclass
class Exercise(ABC):
    name: str
    checker: Path
    timeout: timedelta = timedelta(seconds=15)
    solver: str = "gecode"
    param_file: Optional[Path] = None

    @staticmethod
    def from_dict(exercise: Dict[str, Any], parent: Dict[str, Any], sol_exercise: bool):
        vals = {**parent, **exercise}
        args = {
            k: v
            for k, v in vals.items()
            if k in ["name", "checker", "objective", "solver", "UNSAT"]
        }
        root = Path(parent.get("root", "."))
        args["checker"] = (root / args["checker"]).absolute()
        if "timeout" in vals:
            args["timeout"] = timedelta(seconds=vals["timeout"])
        if "param_file" in vals:
            args["param_file"] = (root / vals["param_file"]).absolute()

        if sol_exercise:
            if "data" in vals:
                args["data"] = (root / vals["data"]).absolute()
            if "thresholds" in vals:
                args["thresholds"] = [float(j) for j in vals["thresholds"]]

            return SolutionExercise(**args)
        else:
            args["instances"] = []
            for inst in vals["instances"]:
                nargs = inst.copy()
                if "data" in nargs:
                    nargs["data"] = (root / inst["data"]).absolute()
                if "thresholds" in nargs:
                    nargs["thresholds"] = [float(j) for j in inst["thresholds"]]
                args["instances"].append(ModelInstance(**nargs))

            return ModelExercise(**args)

    def run_checker(
        self, submission: str, data: Optional[Path], thresholds: Optional[List[float]]
    ) -> Dict[str, Any]:
        logging.info(f"Run {self.checker} with solution data:\n{submission}")
        # Check whether submission is JSON
        is_json = False
        try:
            _ = json.loads(submission)
            is_json = True
        except JSONDecodeError:
            pass

        solver = minizinc.Solver.lookup("gecode")
        with NamedTemporaryFile(
            prefix="submission", suffix=".json" if is_json else ".dzn"
        ) as temp:
            solution = Path(temp.name)
            solution.write_text(submission)

            instance = minizinc.Instance(solver)
            if self.param_file is not None:
                instance.add_file(self.param_file)
            instance.add_file(self.checker)
            if data is not None:
                instance.add_file(data, parse_data=False)
            instance["thresholds"] = thresholds if thresholds is not None else []
            instance.add_file(solution, parse_data=False)

            result = instance.solve(timeout=self.timeout)
            assert result.status in [
                minizinc.Status.SATISFIED,
                minizinc.Status.ALL_SOLUTIONS,
                minizinc.Status.OPTIMAL_SOLUTION,
            ]

            logging.debug(f"Checker output:\n{str(result)}")

            return json.loads(str(result))

    @abstractmethod
    def grade(self, submission: Path) -> Feedback:
        pass


@dataclass
class ModelInstance:
    UNSAT: bool = False
    data: Optional[Path] = None
    thresholds: Optional[List[float]] = None


@dataclass
class SolutionExercise(ModelInstance, Exercise):
    def grade(self, submission: Path) -> Feedback:
        logging.info(f"Grading solution exercise `{self.name}`")
        raw = submission.read_bytes()

        # Check status
        status = minizinc.Status.from_output(raw, minizinc.Method.MAXIMIZE)
        if status is minizinc.Status.ERROR:
            logging.error(f"Submission contained the ERROR status")
            # Workaround for MiniZinc <= 2.5.5 on Windows giving ERROR status on timeout
            if b"----------" not in raw:
                return Feedback(feedback=SOLUTION_ERROR)
        elif status in [minizinc.Status.UNBOUNDED, minizinc.Status.UNSATISFIABLE]:
            logging.error(f"Submission contained the UNSAT/UNBOUNDED status")
            if self.UNSAT:
                return Feedback(
                    fractionalScore=1.0,
                    feedback=UNSAT_MSG,
                )
            else:
                return Feedback(feedback=UNSAT_ERROR)
        elif status is minizinc.Status.UNKNOWN:
            logging.error(f"Submission contained the UNKNOWN status")
            return Feedback(
                feedback=UNKNOWN_MSG,
            )

        logging.info(f"Submission contained the {status} status")
        # Split solutions
        raw = re.sub(rb"^\w*%.*\n?", b"", raw, flags=re.MULTILINE)
        raw = re.sub(
            rb"={5}(ERROR|UNKNOWN|UNSATISFIABLE|UNSATorUNBOUNDED|UNBOUNDED|)?={5}",
            b"",
            raw,
        )
        solutions = [
            sol.strip() for sol in raw.split(b"----------") if sol.strip() != b""
        ]
        if len(solutions) < 1:
            return Feedback(
                feedback=EMPTY_ERROR,
            )

        try:
            result = self.run_checker(
                solutions[-1].decode(), self.data, self.thresholds
            )
        except minizinc.MiniZincError as err:
            logging.error(f"An error occurred while running the checker:\n{err}")
            return Feedback(
                feedback=INPUT_ERROR,
            )

        assert not (result["correct"] and self.UNSAT), GRADER_LAPSE
        return Feedback.from_dict(result)


@dataclass
class ModelExercise(Exercise):
    instances: List[ModelInstance] = field(default_factory=list)

    def grade(self, submission: Path) -> Feedback:
        logging.info(f"Grading model exercise `{self.name}`")

        has_statistics_checker = self.has_statistics_checker()
        if has_statistics_checker:
            logging.info("Checker has statistics checker")

        with NamedTemporaryFile(
            prefix="submission", suffix=".mzn", dir=self.checker.parent
        ) as temp:
            model_file = Path(temp.name)
            model_file.write_bytes(submission.read_bytes())

            solver = minizinc.Solver.lookup(self.solver)
            try:
                model = minizinc.Model(model_file)
                if self.param_file is not None:
                    model.add_file(self.param_file)

                instance = minizinc.Instance(solver, model)
            except minizinc.MiniZincError as err:
                logging.error(
                    f"An error occurred while running the model submission:\n{err}"
                )
                return Feedback(feedback=MODEL_ERROR)
            assert isinstance(instance, minizinc.CLI.CLIInstance)

            scores: List[float] = []
            feedback: List[str] = []
            for inst in self.instances:
                try:
                    with instance.branch() as child:
                        if inst.data is not None:
                            child.add_file(inst.data, parse_data=False)

                        child.add_file(self.checker)
                        child.add_string("array[int] of float: thresholds;")
                        child["thresholds"] = (
                            inst.thresholds if inst.thresholds is not None else []
                        )

                        logging.info(
                            f"Running submitted model with data file `{inst.data}`"
                        )
                        result = child.solve(
                            timeout=self.timeout, intermediate_solutions=True
                        )
                except minizinc.MiniZincError as err:
                    logging.error(
                        f"An error occurred while running the model submission:\n{err}"
                    )
                    scores.append(0.0)
                    feedback.append(MODEL_ERROR)
                    continue

                if result.status is minizinc.Status.ERROR:
                    logging.error(
                        f"Submission with {inst.data} contained the ERROR status"
                    )
                    scores.append(0.0)
                    feedback.append(MODEL_ERROR)
                elif result.status in [
                    minizinc.Status.UNBOUNDED,
                    minizinc.Status.UNSATISFIABLE,
                ]:
                    logging.error(
                        f"Submission with {inst.data} returned the UNSAT/UNBOUNDED status"
                    )
                    if inst.UNSAT:
                        scores.append(1.0)
                        feedback.append(UNSAT_MSG)
                    else:
                        scores.append(0.0)
                        feedback.append(UNSAT_ERROR)
                elif result.status is minizinc.Status.UNKNOWN:
                    logging.error(
                        f"Submission with {inst.data} returned the UNKNOWN status"
                    )
                    scores.append(0.0)
                    feedback.append(UNKNOWN_MSG)
                else:
                    logging.info(
                        f"Submission with {inst.data} returned the {result.status} status"
                    )
                    try:
                        for solution in result.solution:
                            logging.debug(f"Checker output:\n{solution.check()}")
                            checked = json.loads(solution.check())

                            if not checked.get("correct", True):
                                logging.warning(f"Solution checker reported errors!")
                                break
                            else:
                                assert not inst.UNSAT, GRADER_LAPSE
                    except (minizinc.MiniZincError, JSONDecodeError) as err:
                        logging.error(
                            f"An error occurred while running the checker:\n{err}"
                        )
                        scores.append(0.0)
                        feedback.append(INPUT_ERROR)
                        continue
                    if has_statistics_checker and checked.get("correct", True):
                        # Use final statistics check
                        assert (
                            "statisticsCheck" in result.statistics
                        ), GRADER_CHECKER_LAPSE
                        stat_check = result.statistics["statisticsCheck"]
                        logging.debug(f"Statistics check output:\n{stat_check}")
                        checked = json.loads(json.loads('"' + stat_check + '"'))
                    scores.append(checked["fractionalScore"])
                    feedback.append(checked["feedback"])

        feedback_str = "\n".join(
            [
                "#### "
                + (
                    self.instances[i].data.name.upper()
                    if self.instances[i].data is not None
                    else self.name.upper()
                )
                + " - "
                + str(int(scores[i] * 100))
                + "% ####\n"
                + feedback[i]
                + "\n"
                for i in range(len(scores))
            ]
        )
        return Feedback(fractionalScore=mean(scores), feedback=feedback_str)

    def has_statistics_checker(self) -> bool:
        model = minizinc.Model()
        checker = Path(self.checker)
        text = checker.read_text()
        if self.checker.suffix == ".mzc":
            text = zlib.decompress(base64.b64decode(text)).decode()
        model.add_string(text)
        model.add_string(
            "string: grader_has_statistics_check = checkStatistics(0, 0, 0, 0);"
        )
        try:
            minizinc.Instance(minizinc.Solver.lookup("gecode"), model)
        except minizinc.MiniZincError as err:
            return False
        return True


def lookup_exercise(conf: Path, id: str) -> Optional[Exercise]:
    import yaml

    logging.info(f"Initialising exercise library from {conf}")
    assert conf.exists()
    configuration = yaml.safe_load(conf.read_bytes())

    reset = os.getcwd()
    os.chdir(str(conf.parent))
    try:
        for assignment in configuration:
            for exercise in assignment.get("solution_exercises", []):
                if id == exercise["id"]:
                    return Exercise.from_dict(exercise, assignment, True)

            for exercise in assignment.get("model_exercises", []):
                if id == exercise["id"]:
                    return Exercise.from_dict(exercise, assignment, False)

        return None
    finally:
        os.chdir(reset)


def coursera():
    logging.basicConfig(level=logging.DEBUG)
    logging.info(f"Grader started: {str(sys.argv)}")
    feedback = Feedback()
    try:
        # Default location for the submission to be placed.
        location: Path = Path("/shared/submission/submission.sub")
        # The Coursera partId is set through an environmental variable.
        part_id: str = os.environ["partId"]
        logging.info(f"Submission partId: {part_id}")

        # Lookup exercise in library
        exercise = lookup_exercise(
            Path(os.environ.get("GRADER_LIB", "./assignments.yaml")), part_id
        )
        if exercise is None:
            logging.error(f"Exercise {part_id} could not be located")
        else:
            # Grade assignment
            logging.info(f"Exercise {part_id} parsed as: {exercise}")
            feedback = exercise.grade(location)

    finally:
        logging.info("Output feedback: " + feedback.serialise())
        Path("/shared/feedback.json").write_text(feedback.serialise())
