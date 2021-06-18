import matplotlib.pyplot as plt


def plot_avg_trajectory_len(train_stats, save_path: str = './traj_len_fig.png'):
    x_init = 0
    y_init = train_stats['init_avg_traj_len']
    print(x_init)
    print(y_init)

    x_train = [i+1 for i in range(len(train_stats['train_avg_traj_len']))]
    y_train = train_stats['train_avg_traj_len']
    print(x_train)
    print(y_train)

    x_final = len(train_stats['train_avg_traj_len']) + 1
    y_final = train_stats['final_avg_traj_len']
    print(x_final)
    print(y_final)

    plt.plot(x_init, y_init, 'bo')
    plt.plot(x_train, y_train, 'g-')
    plt.plot(x_final, y_final, 'ro')

    plt.title('Average Trajectory Length vs Number of Epochs')
    plt.xlabel('')
    plt.ylabel('')

    axes = plt.gca()
    axes.yaxis.grid()

    if save_path is not None:
        plt.savefig(save_path,
                    format='png',
                    dpi=100,
                    bbox_inches='tight')

    #plt.show()
