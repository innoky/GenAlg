import re
import sys
from pathlib import Path

PATTERN = re.compile(r"(<!-- BEGIN include: (?P<path>[^ ]+) -->\n)(?P<body>.*?)(<!-- END include -->)",
                     re.S)


def render(report: Path) -> None:
    text = report.read_text(encoding="utf-8")

    def sub(m: re.Match) -> str:
        content = (report.parent / m["path"]).read_text(encoding="utf-8").rstrip() + "\n"
        return m.group(1) + content + m.group(4)

    report.write_text(PATTERN.sub(sub, text), encoding="utf-8")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        render(Path(p))
        print(f"обновлено: {p}")
