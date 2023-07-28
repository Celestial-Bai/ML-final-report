from basicsetting import *
from nn_tr import *


if __name__ == '__main__':
    model = NN()
    sdict = torch.load('checkpoint_nn_v4.pt')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(sdict)
    model = model.to(device)
    test_data = pd.read_csv('test_new.csv')
    test_data = np.array(test_data, dtype=np.float32)
    test_data = torch.tensor(test_data)
    #train_sampler = RandomSampler(test_data)
    X_test_dataloader = DataLoader(test_data, batch_size=128, num_workers=3, shuffle=False)
    preds = torch.empty((0, 1))
    for i, seq in enumerate(X_test_dataloader):
        X_seq = seq
        X_seq = X_seq.to(device)
        pred = model(X_seq)
        pred = pred.detach().cpu()
        preds = torch.cat((preds, pred))
    f = open('nn_mh_wlog.txt', mode='w')
    for i in range(len(preds)):
        print(float(np.exp(preds[i][0])), file=f)
    f.close()
