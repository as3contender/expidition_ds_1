import subprocess, sys, yaml
from pathlib import Path


def run(cmd: list[str]):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main(cfg="configs/default.yaml"):
    # 1) индекс и сплиты
    run(
        [
            sys.executable,
            "-m",
            "src.build_splits",
        ]
    )
    # 2) обучение
    run([sys.executable, "-m", "src.train"])
    # 3) инференс и сабмит
    run([sys.executable, "-m", "src.infer_to_geojson"])


if __name__ == "__main__":
    main()
