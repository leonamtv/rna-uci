import os

def filter_dataset ( file_path, format ) :

    if not os.path.isfile ( file_path ) :
        raise Exception ( f"Arquivo de dados não encontrado: { file_path }" )

    data = []

    with open(file_path, 'r') as file :
        for line in file :
            if '?' not in line :
                full_line    = [ float(item) for item in line.split(',') ]
                input_entry  = full_line[:format['input_size']]
                output_entry = full_line[(format['input_size']):]
                data.append(( input_entry, output_entry ))

    return data

    