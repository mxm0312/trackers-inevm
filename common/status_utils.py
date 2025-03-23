from typing import Dict, List


def generate_progress_data(
    progress: float, stage: str, statistics: Dict = None
) -> Dict:
    """
    {
        "stage": <номер этапа обработки и сколько всего этапов, например "1 из 3">
        "progress": <процент выполнения обработки на данном этапе>,
        "statistics": { <объект статистических данных, например, со следующими атрибутами:>
                "out_file": <имя выходного файла с разметкой>,
                "chains_count": <количество рассчитанных цепочек в файле>,
                "markups_count": <количество рассчитанных примитивов в файле>,
                "train_error": <ошибка на обучающих данных>,
                "test_error": <ошибка на тестовых данных>
            }
    }
    """
    data = {
        "stage": stage,
        "progress": progress,
    }
    if statistics is not None:
        data["statistics"] = statistics
    return data


def generate_statistics(
    out_file: str,
    chains_count: int,
    markups_count: int,
    train_error=None,
    test_error=None,
    verbose=False,
):
    data = {
        "out_file": out_file,
        "chains_count": chains_count,
        "markups_count": markups_count,
        "train_error": train_error,
        "test_error": test_error,
    }
    if train_error is not None:
        data["train_error"] = train_error
    if test_error is not None:
        data["test_error"] = test_error
    if verbose:
        print(
            f"output file saved to {out_file}/nchain_count={chains_count}/nmarkups_count={markups_count}/n"
        )
    return data


def count_markups(file_markup: Dict):
    total_markups = 0
    for chain in file_markup["file_chains"]:
        total_markups += len(chain["chain_markups"])
    return total_markups


def generate_before_end(
    out_files: List[str], chains_count: List[int], markups_count: List[int]
):
    return {
        "out_files": out_files,
        "chains_count": chains_count,
        "markups_count": markups_count,
    }


def generate_error_data(msg: str, details: str) -> Dict:
    return {"msg": msg, "details": details}
