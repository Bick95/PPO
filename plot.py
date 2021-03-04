import matplotlib.pyplot as plt


def plot_avg_trajectory_len(train_stats):

    x_train = [i+1 for i in range(len(train_stats['train_avg_traj_len']))]
    x_non_train = [0] + [len(train_stats['train_avg_traj_len']) + 1]

    # Creating plot with dataset_1
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Average Trajectory Length in Time Steps', color=color)
    ax1.plot(x_train, train_stats['train_avg_traj_len'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel('Y2-axis', color=color)
    ax2.plot(x_non_train, [train_stats['init_avg_traj_len'], train_stats['final_avg_traj_len']], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    #print('Plotting time')

    #x_init = 0
    #y_init = train_stats['init_avg_traj_len']
    #print(x_init)
    #print(y_init)

    #x_train = [0] + [i+1 for i in range(len(train_stats['train_avg_traj_len']))] + [len(train_stats['train_avg_traj_len']) + 1]
    #y_train = train_stats['train_avg_traj_len']
    #print(x_train)
    #print(y_train)

    #x_final = len(train_stats['train_avg_traj_len']) + 1
    #y_final = train_stats['final_avg_traj_len']
    #print(x_final)
    #print(y_final)

    #plt.plot(x_init, y_init, 'bo')
    #plt.plot(x_train, y_train, 'g-')
    #plt.plot(x_final, y_final, 'ro')

    #plt.title('Average Trajectory Length vs Number of Epochs')
    #plt.xlabel('')
    #plt.ylabel('')

    axes = plt.gca()
    axes.yaxis.grid()

    plt.show()

    print('End')
