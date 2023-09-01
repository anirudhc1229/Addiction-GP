import random
import statistics
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    df = pd.read_csv('smoker_data/data.dat', delimiter='\t')

    df.columns = ['id', 'day', 'week', 'dotw', 
                    'subj', 'gender', 'age', 
                    'mon', 'tue', 'wed', 'thu', 'fri', 'sat',
                    'y', 'y1', 'y2', 'y3', 'y4', 'y7', 'y8', 'y9', 
                    'y14', 'y15', 'y16', 'y21', 'y28', 
                    'y35', 'y42', 'y49', 'y56', 'y63']
    
    n_subj = 75
    data = [[] for _ in range(n_subj)]

    for i in range(len(df.index)):
        if df.loc[i]['subj'] == -9:
            continue
        data[int(df.loc[i]['subj']) - 1].append(df.loc[i]['y'])

    # clean up missing data
    for i in range(len(data)):
        if -9 in data[i]:
            data[i] = data[i][:data[i].index(-9)]
    data = [d for d in data if len(d) != 0]

    # for i in range(len(data[:2])):
    #     t = range(len(data[i]))
    #     plt.plot(t, data[i])
    # plt.show()

    # approximate original data
    for i in range(len(data)):
        mean = statistics.mean(data[i])
        sd = 0.1361 * mean + 1.436 # linear regression from mean,sd pairs in paper
        for j in range(len(data[i])):
           data[i][j] *= sd
           data[i][j] = round(data[i][j])

    for i in range(len(data[:2])):
        t = range(len(data[i]))
        plt.plot(t, data[i])
    plt.show()

    # print(data)

    with open('smoker_data/cleaned_data.txt', 'w') as f:
        f.write('"st","period","mean","std","x","xx"')
        for i in range(len(data)):
            f.write('\n')
            f.write('"' + str(i) + '",')
            f.write('"DAILY",')
            f.write(str(statistics.mean(data[i])) + ',')
            f.write(str(statistics.stdev(data[i])) + ',')
            f.write('"')
            spl = len(data[i]) * 2 // 3
            for j in range(len(data[i][:spl])):
                f.write(str(data[i][j]))
                if j != spl - 1:
                    f.write(';')
            f.write('",')
            f.write('"')
            for j in range(spl, len(data[i])):
                f.write(str(data[i][j]))
                if j != len(data[i]) - 1:
                    f.write(';')
            f.write('"')
