import matplotlib.pyplot as plt
def show_train_history(log_train,log_test,name):
    plt.figure().clear()
    plt.plot(log_train)
    plt.plot(log_test)
    plt.title('Training History')
    plt.ylabel(name)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('plot/{}.png'.format(name))
    plt.close()