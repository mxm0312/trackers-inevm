from typing import Dict


def generate_progress_data(
    progress: float, stage: int, output_file: str = None
) -> Dict:
    data = {
        "on_progress": progress,
        "stage": stage,
    }
    if output_file is not None:
        data["statistics"] = {"out_file": output_file}
    return data

def generate_error_data(msg: str) -> Dict:
    return {'on_error':msg}