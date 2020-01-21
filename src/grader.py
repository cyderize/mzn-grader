#!/usr/bin/env python3
import json
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple

from minizinc import Solver, Instance, Status, MiniZincError, Method, Model

ERROR = (
    "An error occurred within the grader, please inform your course "
    "instructor.\n\n The course instructor will need to for which "
    "assignment the error occurred, and at what time you made your "
    "submission. Thank you for your help and your patience. We hope to "
    "prevent these issues from happening in the future."
)


@dataclass
class Feedback:
    fractionalScore: float = 0.0
    feedback: str = ERROR

    def serialise(self):
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

    @staticmethod
    def from_dict(exercise: Dict[str, Any], parent: Dict[str, Any], sol_exercise: bool):
        vals = {**parent, **exercise}
        args = {
            k: v
            for k, v in vals.items()
            if k in ["name", "checker", "objective", "solver"]
        }
        root = Path(parent.get("root", "."))
        args["checker"] = (root / args["checker"]).absolute()
        if "timeout" in vals:
            args["timeout"] = timedelta(milliseconds=vals["timeout"])

        if sol_exercise:
            if "data" in vals:
                args["data"] = (root / vals["data"]).absolute()
            if "thresholds" in vals:
                args["thresholds"] = [float(j) for j in vals["thresholds"]]

            return SolutionExercise(**args)
        else:
            args["instance_thresholds"] = [
                (
                    (root / vals["data"][j]).absolute(),
                    [float(k) for k in vals["thresholds"][j]],
                )
                for j in range(len(vals["data"]))
            ]

            return ModelExercise(**args)

    def run_checker(
        self, submission: str, data: Optional[Path], thresholds: List[float]
    ) -> Dict[str, Any]:
        logging.info(f"Run {self.checker} with solution data:\n{submission}")
        solver = Solver.lookup(self.solver)
        with NamedTemporaryFile(prefix="submission", suffix=".dzn") as temp:
            solution = Path(temp.name)
            solution.write_text(submission)

            instance = Instance(solver)
            instance.add_file(self.checker)
            if data is not None:
                instance.add_file(data, parse_data=False)
            if len(thresholds) > 0:
                instance["thresholds"] = thresholds
            instance.add_file(solution, parse_data=False)

            result = instance.solve(timeout=self.timeout)
            assert result.status in [
                Status.SATISFIED,
                Status.ALL_SOLUTIONS,
                Status.OPTIMAL_SOLUTION,
            ]

            logging.debug(f"Checker output:\n{str(result)}")

            return json.loads(str(result))

    @abstractmethod
    def grade(self, submission: Path) -> Feedback:
        pass


@dataclass
class SolutionExercise(Exercise):
    data: Optional[Path] = None
    thresholds: List[float] = field(default_factory=list)

    def grade(self, submission: Path) -> Feedback:
        logging.info(f"Grading solution exercise `{self.name}`")
        raw = submission.read_bytes()

        # Check status
        status = Status.from_output(raw, Method.MAXIMIZE)
        if status is Status.ERROR:
            logging.error(f"Submission contained the ERROR status")
            return Feedback(
                feedback=(
                    "An error occurred while solving your model.\n\nEnsure that your "
                    "model does not contain any elements that are not supported by "
                    "the solver and check that your model returns no error message "
                    "when running locally. If the problem persists, then please ask "
                    "your course instructor for help."
                )
            )
        elif status in [Status.UNBOUNDED, Status.UNSATISFIABLE]:
            logging.error(f"Submission contained the UNSAT/UNBOUNDED status")
            return Feedback(
                feedback=(
                    "Your model reported the problem as unsatisfiable, but the "
                    "problem is satisfiable.\n\nPlease ensure that your model "
                    "contains only the constraints that are part of the model "
                    "description."
                )
            )
        elif status is Status.UNKNOWN:
            logging.error(f"Submission contained the UNKNOWN status")
            return Feedback(
                feedback=(
                    "Your submission is unable to find a feasible "
                    "solution to the problem within the set time limit."
                ),
            )

        # Split solutions
        raw = re.sub(rb"^\w*%.*\n?", b"", raw, flags=re.MULTILINE)
        raw = re.sub(rb"=====[^=]*=====", b"", raw)
        solutions = [
            sol.strip() for sol in raw.split(b"----------") if sol.strip() != b""
        ]
        assert len(solutions) >= 1

        try:
            result = self.run_checker(
                solutions[-1].decode(), self.data, self.thresholds
            )
        except MiniZincError as err:
            logging.error(f"An error occurred while running the checker:\n{err}")
            return Feedback(
                feedback=(
                    "An error occurred while checking your solution.\n\nCheck your "
                    "output statement and make sure it meets the requirements of the "
                    "assignment. If the problem persists, then please ask your course "
                    "instructor for help."
                ),
            )

        return Feedback.from_dict(result)


@dataclass
class ModelExercise(Exercise):
    instance_thresholds: List[Tuple[Path, List[float]]] = field(default_factory=list)
    timeout: timedelta = timedelta(seconds=60)

    def grade(self, submission: Path) -> Feedback:
        logging.info(f"Grading model exercise `{self.name}`")
        with NamedTemporaryFile(prefix="submission", suffix=".mzn") as temp:
            model = Path(temp.name)
            model.write_bytes(submission.read_bytes())

            solver = Solver.lookup(self.solver)
            instance = Instance(solver, Model(model))

            scores: List[float] = []
            feedback: List[str] = []
            for (data, thresholds) in self.instance_thresholds:
                with instance.branch() as child:
                    child.add_file(data, parse_data=False)
                    try:
                        logging.info(f"Running submitted model with data file `{data}`")
                        result = child.solve(timeout=self.timeout)
                    except MiniZincError as err:
                        logging.error(
                            f"An error occurred while running the model submission:\n {err}"
                        )
                        return Feedback(
                            feedback=(
                                "An error occurred while solving your "
                                "model.\n\nPlease ensure that your MiniZinc model "
                                "compiles correctly and works for all provided "
                                "instances. If the problem persists, then please ask "
                                "your course instructor for help."
                            ),
                        )

                if result.status is Status.ERROR:
                    logging.error(f"Submission with {data} contained the ERROR status")
                    return Feedback(
                        feedback=(
                            "An error occurred while solving your model.\n\nEnsure "
                            "that your model does not contain any elements that are "
                            "not supported by the solver and check that your model "
                            "returns no error message when running locally. If the "
                            "problem persists, then please ask your course instructor "
                            "for help."
                        )
                    )
                elif result.status in [Status.UNBOUNDED, Status.UNSATISFIABLE]:
                    logging.error(
                        f"Submission with {data} returned the UNSAT/UNBOUNDED status"
                    )
                    return Feedback(
                        feedback=(
                            "Your model reported the problem as unsatisfiable, but the "
                            "problem is satisfiable.\n\nPlease ensure that your model "
                            "contains only the constraints that are part of the model "
                            "description."
                        )
                    )
                elif result.status is Status.UNKNOWN:
                    logging.error(f"Submission with {data} returned the UNKNOWN status")
                    scores.append(0.0)
                    feedback.append(
                        "Your submission is unable to find a feasible solution to the "
                        "problem within the set time limit."
                    )
                else:
                    checked = self.run_checker(str(result), data, thresholds)
                    if not checked["correct"]:
                        logging.warning(f"Solution checker reported errors! Stop grading")
                        return Feedback.from_dict(checked)
                    else:
                        scores.append(checked["fractionalScore"])
                        feedback.append(checked["feedback"])

        feedback_str = "\n".join(
            [
                "#### "
                + self.instance_thresholds[i][0].name.upper()
                + " - "
                + str(int(scores[i] * 100))
                + "% ####\n"
                + feedback[i]
                + "\n"
                for i in range(len(scores))
            ]
        )
        return Feedback(fractionalScore=mean(scores), feedback=feedback_str)


def lookup_exercise(conf: Path, id: str) -> Optional[Exercise]:
    import yaml

    logging.info(f"Initialising exercise library from {conf}")
    assert conf.exists()
    configuration = yaml.safe_load(conf.read_bytes())

    reset = os.curdir
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info(f"Grader started: {str(sys.argv)}")
    feedback = Feedback()
    try:
        # Initialise grader from command line (Coursera submission format)
        user: str = ""
        filename: str = ""
        location: Path = Path("/shared/submission/submission.sub")
        partID: str

        for i in range(len(sys.argv)):
            if sys.argv[i] == "userId":
                user = sys.argv[i + 1]
            elif sys.argv[i] == "filename":
                filename = sys.argv[i + 1]
            elif sys.argv[i] == "override_sub":
                location = Path(sys.argv[i + 1])
            elif sys.argv[i] == "partId":
                partID = sys.argv[i + 1]

        logging.info("Submission information:")
        logging.info(f"> User: {user}")
        logging.info(f"> Filename: {filename}")
        logging.info(f"> Location: {location}")
        logging.info(f"> partID: {partID}")

        # Lookup exercise in library
        exercise = lookup_exercise(
            Path(os.environ.get("GRADER_LIB", "./assignments.yaml")), partID
        )
        if exercise is None:
            logging.error(f"Exercise {partID} could not be located")
        else:
            # Grade assignment
            logging.info(f"Exercise {partID} parsed as: {exercise}")
            feedback = exercise.grade(location)

    finally:
        logging.info("Output feedback: " + feedback.serialise())
        print(feedback.serialise())
