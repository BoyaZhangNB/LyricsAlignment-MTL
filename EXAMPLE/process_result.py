import csv
from pathlib import Path
from typing import List

def csv_differences(csv_path: str) -> List[float]:
    """
    Read a CSV whose rows look like:  float1, float2, str
    and return a list containing:
        [float2 - float1,              # first row
         float4 - float2,              # second row’s float2 minus first row’s float2
         float6 - float4, …]           # and so on

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file.

    Returns
    -------
    List[float]
        List of integer differences, as specified above.
    """
    deltas: List[float] = []
    previous_second: float | None = None

    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:                # skip blank lines
                continue
            first, second = map(float, row[:2])  # cast the first two columns to float

            if previous_second is None:
                # For the very first row: float2 - float1
                deltas.append(second - first)
            else:
                # For subsequent rows: current float2 - previous row's float2
                deltas.append(second - previous_second)

            previous_second = second   # update for next iteration

    return deltas

if __name__ == "__main__":
    result = csv_differences("EXAMPLE/MTL.csv")
    print(result)
    with open("result.txt", "w") as f:
        result = str(result)
        f.write(result)
