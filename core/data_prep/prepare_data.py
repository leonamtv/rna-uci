import os
import matplotlib.pyplot as plt

def filter_dataset ( file_path, format, normalize=False, visualize_distribution=False ) :

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

    if normalize :
        min_array = [ float('inf') for _ in range(format['input_size']) ]
        max_array = [ 0 for _ in range(format['input_size']) ]

        for sample in data :
            for i, inp in enumerate(sample[0]) :
                if inp > max_array[i] :
                    max_array[i] = inp
                if inp < min_array[i] :
                    min_array[i] = inp

        if visualize_distribution :
            inputs = [ [] for _ in range(format['input_size']) ]

        for sample in data :
            for i in range(len(sample[0])) :
                sample[0][i] = sample[0][i] / float( max_array[i] - min_array[i] )
                if visualize_distribution :
                    inputs[i].append(sample[0][i])

        if visualize_distribution :
            for i, inp in enumerate(inputs) :
                plt.figure()
                plt.title(f'Histograma input {i + 1}')
                plt.hist(x=inp, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
                plt.grid(axis='y')
                plt.show()

        return data

    else :

        return data
        

    