import pandas as pd
import numpy


def read_fasta_data(file: str) -> pd.DataFrame:
    names = []
    seq = []
    year = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                names.append(line[1:])
                if line[line.rfind('_') + 1:].isdigit():
                    year.append(int(line[line.rfind('_') + 1:]))
                else:
                    year.append(None)
            else:
                seq.append(line)
    data = {
        'Name': names,
        'Year': year,
        'Sequence': seq
    }
    df = pd.DataFrame(data)  
    return df