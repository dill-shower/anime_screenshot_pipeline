import subprocess
import os
import sys


def run(command, desc=None):
    if desc is not None:
        print(desc)

    # Join the command list into a single string if it's a list
    if isinstance(command, list):
        command = " ".join(command)

    process = subprocess.run(command, shell=True, capture_output=False)
    if process.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error code: {process.returncode}")
        sys.exit(1)


def install_package(package, command):
    if not is_installed(package):
        run(command, f"Installing {package}")


def is_installed(package):
    try:
        subprocess.run(
            f"uv pip show {package} --python {sys.executable}",
            shell=True,
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_uv_installed():
    """Проверяет наличие uv и устанавливает его при необходимости"""
    try:
        subprocess.run(
            "uv --version",
            shell=True,
            capture_output=True,
            check=True,
        )
        print("uv is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing uv...")
        # Устанавливаем uv через pip (один раз)
        run(
            f"{sys.executable} -m pip install uv",
            "Installing uv package manager",
        )


def prepare_environment():
    # Сначала убедимся что uv установлен
    ensure_uv_installed()

    # Install PyTorch
    # Use cuda 12.1 here for consistency with onnxruntime but that
    # does not really matter since they use cuda from different places anyway
    install_package(
        "torch",
        (
            f"uv pip install torch torchvision torchaudio "
            f"--index-url https://download.pytorch.org/whl/cu128 "
            f"--python {sys.executable}"
        ),
    )

    # Install other requirements from requirements.txt
    requirements_path = os.path.join(os.getcwd(), "requirements.txt")
    if os.path.exists(requirements_path):
        run(
            f"uv pip install -r {requirements_path} --python {sys.executable}",
            "Installing requirements from requirements.txt",
        )

    # Install waifuc package
    waifuc_path = os.path.join(os.getcwd(), "waifuc")
    if os.path.exists(waifuc_path):
        os.chdir(waifuc_path)
        run(
            f"uv pip install . --python {sys.executable}",
            "Installing waifuc package",
        )


if __name__ == "__main__":
    prepare_environment()
