import textwrap


class OperatorIssue(RuntimeError):
    @staticmethod
    def operator_str(target, args, kwargs):
        lines = [f"target: {target}"] + [
            f"args[{i}]: {arg}" for i, arg in enumerate(args)
        ]
        if kwargs:
            lines.append(f"kwargs: {kwargs}")
        return textwrap.indent("\n".join(lines), "  ")


class MissingOperator(OperatorIssue):
    def __init__(self, target, args, kwargs):
        # with open("/tmp/missing_ops.txt", "a") as fd:
        #    fd.write(str(target) + "\n")
        super().__init__(
            f"missing lowering/decomposition\n{self.operator_str(target, args, kwargs)}"
        )


class LoweringException(OperatorIssue):
    def __init__(self, exc, target, args, kwargs):
        super().__init__(
            f"{type(exc).__name__}: {exc}\n{self.operator_str(target, args, kwargs)}"
        )


class InvalidCxxCompiler(RuntimeError):
    def __init__(self):
        from . import config

        super().__init__(
            f"No working C++ compiler found in {config.__name__}.cpp.cxx: {config.cpp.cxx}"
        )


class CppCompileError(RuntimeError):
    def __init__(self, cmd, output):
        super().__init__(
            textwrap.dedent(
                """
                C++ Compiler Failed

                Command:
                {0}

                Output:
                {1}
                """
            )
            .strip()
            .format(" ".join(cmd), output.decode("utf-8"))
        )
