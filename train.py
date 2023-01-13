import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from data import DatasetOnlineGen
from model import LELAGNN
from loss import  BCEMask
import numpy as np
from data import SytheticValidation
import os
from torch.utils.tensorboard import SummaryWriter


for i_run in range(10): #train LELA model 10 runs
    device = "cuda:0"
    net = LELAGNN()

    optimizer = optim.Adam(
        net.parameters(),
        amsgrad=True,
    )
    net.to(device)

    criterion = BCEMask()

    dataset = DatasetOnlineGen(
        size=100000, # this is just to trick data loader to work, the dataset is generated on the fly so there is no dataset size
        max_n_lfs=60,
        max_example=2000,
    )


    valid = SytheticValidation()# the sythetic validation set


    collate_fn = dataset.collate

    dataloader = DataLoader(
        dataset,
        batch_size=50,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    writer = SummaryWriter(comment="run_"+str(i_run))
    n_iter = 0
    optimizer.zero_grad()
    n_not_improved = 0
    min_loss = np.inf
    losses = []
    val_accs = []
    for _ in range(1000000):  
        if n_not_improved > 10**4:
            break
        for i, (index, value, labels) in enumerate(dataloader):
            # Place tensors on GPU
            index = index.int()
            index, value, labels = index.squeeze().to(
                device), value.squeeze().to(device), labels.squeeze().to(device)

            outputs = net(index, value)

            outputs, mask = outputs
            loss = criterion(outputs, labels.float(), mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            l_value = loss.item()

            n_iter += 1

            eval_fre = 100
            losses.append(l_value)
            if n_iter%eval_fre==0:
                net.eval()
                with torch.no_grad():
                    test_score_sythetic_ind = valid.get_avg_score_sythetic(
                        net)
                writer.add_scalar('score/test_syth_acc_ind',
                                    test_score_sythetic_ind, n_iter)
                val_accs.append(test_score_sythetic_ind)
                l_avg = np.mean(losses[-1000:])
                writer.add_scalar('score/train_loss_iter',l_avg, n_iter)
                if l_avg< min_loss:
                    min_loss = l_avg
                    n_not_improved = 0
                else:
                    n_not_improved+=eval_fre
                net.train()
        if not os.path.exists("model_checkpoints"):
            os.mkdir("model_checkpoints")

        torch.save(
            {
                'n_iter': n_iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_avg':np.mean(val_accs),
            },
            "model_checkpoints/model_" + str(i_run) + ".pt"
            )

#select the best run
val_accs = []
for i in range(10): 
    checkpoint = torch.load("model_checkpoints/model_"+str(i)+".pt", map_location="cpu")
    val_accs.append(checkpoint['val_acc_avg'])
best_run = np.argmax(val_accs)
print('Please use the checkpoint:', "model_" + str(best_run) + ".pt")
