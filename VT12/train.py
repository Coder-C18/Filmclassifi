import torch.nn as nn
import torch
from dataloader import LoadData
import config
from model import ClassifyFilm
from tqdm import tqdm
from torchmetrics.classification import MultilabelAUROC, MulticlassAccuracy
from sklearn.metrics import accuracy_score,roc_auc_score
from plot_log import show_train_history

train_dataset = LoadData(data_path='data/train')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=True)
test_dataset = LoadData(data_path='data/test')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=True)
model = ClassifyFilm().to(config.device)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)


model.apply(init_normal)

criterion1 = nn.BCELoss()
criterion2 = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
auroc = MultilabelAUROC(num_labels=23, average="macro", thresholds=None).to(config.device)

last_loss = 100
patience = 30
triggertimes = 0
flag = False
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)


log_loss_train = []
log_loss_val = []

log_acc_train = []
log_acc_val = []

log_auc_train = []
log_auc_val = []

for epoch in range(config.epochs):


    train = tqdm(train_loader)

    for i, (images, actor, director, description, country, label) in enumerate(train):
        model.train()
        output1, output2 = model(images, actor, director, description)
        loss1 = criterion1(output1, label)
        loss2 = criterion2(output2, country)
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        acc_train = float(accuracy_score(torch.argmax(output2, dim=1).tolist(), torch.argmax(country, dim=1).tolist()))
        auc_train = float(auroc(output1, label.int()))

        auc_test = 0
        acc_test = 0
        val_loss = 0
        model.eval()

        for images, actor, director, description, country, labels in test_loader:

            pred1, pred2 = model(images, actor, director, description)
            val_loss += criterion1(pred1, labels) + criterion2(pred2, country)

            pred1 = pred1.cpu().detach().numpy()
            auc_test += float(auroc(torch.from_numpy(pred1).to(config.device), labels.int()))
            acc_test += float(
                accuracy_score(torch.argmax(pred2, dim=1).tolist(), torch.argmax(country, dim=1).tolist()))

        val_loss = float(val_loss / len(test_loader))
        auc_test = float(auc_test / len(test_loader))
        acc_test = float(acc_test / len(test_loader))

        train.set_description("Epoch {} / {}".format(epoch, config.epochs))

        log_loss_train.append(float(loss))
        log_loss_val.append(val_loss)
        log_acc_train.append(auc_train)
        log_acc_val.append(acc_test)
        log_auc_train.append(auc_train)
        log_auc_val.append(auc_test)

        train.set_postfix(acc_test=acc_test,
                          auc_test=auc_test,
                          val_loss=val_loss,
                          acc_train=acc_train,
                          auc_train=auc_train,
                          loss_train=float(loss), )

    if val_loss > last_loss:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!\n.')
            torch.save(model.state_dict(), 'model_mlp.ckpt')
            break

    else:
        trigger_times = 0

        last_loss = val_loss
show_train_history(log_train=log_loss_train, log_test=log_loss_val, name='loss')
show_train_history(log_train=log_acc_train, log_test=log_acc_val, name='accuracy')
show_train_history(log_train=log_auc_train, log_test=log_auc_val, name='auc_roc')