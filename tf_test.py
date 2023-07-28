from basicsetting import *
from tf_tr import *


if __name__ == '__main__':
    model = Transformer()
    sdict = torch.load('checkpoint_tf_v8.pt')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(sdict)
    model = model.to(device)
    test_data = pd.read_csv('test_new.csv')
    test_data = np.array(test_data, dtype=np.float32)
    test_data = torch.tensor(test_data)
    #train_sampler = RandomSampler(test_data)
    X_test_dataloader = DataLoader(test_data, batch_size=64, num_workers=3, shuffle=False)
    preds = torch.empty((0, 1))
    for i, seq in enumerate(X_test_dataloader):
        X_seq = seq
        X_seq = X_seq.unsqueeze(1)
        X_seq = X_seq.to(device)
        pred = model(X_seq)
        pred = pred.detach().cpu()
        preds = torch.cat((preds, pred))
    f = open('tf_mh_wlog.txt', mode='w')
    for i in range(len(preds)):
        print(float(np.exp(preds[i][0])), file=f)
    f.close()
