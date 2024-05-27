import csv
import pandas as pd
import os


def eval(eval_paths):
    dict_list = []
    for path in eval_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.fillna(0)
            keys = df['Unnamed: 0'].tolist()
            df = df.to_numpy()

            dict = {}
            for i in range(df.shape[0]):
                key = df[i][0]
                dict[key] = df[i][1:]
            dict_list.append(dict)

    ## Average values across the list
    avg_dict = {}
    for dict in dict_list:
        for key in dict.keys():
            if key in avg_dict.keys():
                avg_dict[key] += dict[key]
            else:
                avg_dict[key] = dict[key]
    avg_dict = {k: v / len(dict_list) for k, v in avg_dict.items()}

    ## Print Avg dict in nice format
    print("Average Evaluation Metrics")
    print("----------------------------")
    for key in avg_dict.keys():
        print(f"{key}: {avg_dict[key][-1]}")


def main():
    object_sequences = ['color1', 'color2', 'color3', 'color4', 'color5']
    subject = ['chandradeep', 'angel', 'zekun2']
    # hand_sequences = ['hand_10', 'hand_10', 'hand_10']
    hand_sequences = ['hand_20', 'hand_20', 'hand_20']

    root_dir = "/users/cpokhari/data/users/cpokhari/neuralgrasp_outputs"

    for i, s in enumerate(subject):
        hand = hand_sequences[i]
        eval_paths = []
        for object in object_sequences:
            name = f'{object}--{hand}'
            print(name)
            path = os.path.join(root_dir, 'composite', s, name, "results/eval_results/eval_metric.csv")
            print(path)
            eval_paths.append(path)

        # print("---------------------------------------------")
        # print(f"Subject: {s}, Hand: {hand}")
        eval(eval_paths)



if __name__ == '__main__':
    main()
