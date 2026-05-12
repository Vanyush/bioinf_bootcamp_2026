import pandas as pd
import numpy


def read_fasta_data(file: str):
    names = []
    seq = []
    year = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                names.append(line[1:])
                if line[line.rfind('_') + 1].isdigit():
                    year.append(int(line[line.rfind('_') + 1:]))
                else:
                    year.append(0)
            else:
                seq.append(line)
    data = {
        'Name': names,
        'Year': year,
        'Sequence': seq
    }
    df = pd.DataFrame(data)  
    return df


df = read_fasta_data('mth_crocuta_aln_renamed.snps.fasta')
df.to_csv('fasta_to_df.csv', index=False)