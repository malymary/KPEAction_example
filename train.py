import os
import argparse
from tqdm import tqdm
import torch
import torch.utils.data.dataloader as DataLoader
import torch.nn as nn



from data import KPEAction
from model import EnDeCoder


def get_args():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--data', default='path/to/data.csv')
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--valrate', default=2)    
    parser.add_argument('--epochs', default=1)
    parser.add_argument('--loss', default="")
    parser.add_argument('--transforms', default=[""])
    
    parser.add_argument('--optimizer', default="adam")

    args = parser.parse_args()
    return args


def log(args, data):
    log_file = os.path.join(args.root, "log.txt")
    print(data)
    with open(log_file, "w") as f:
        f.write(data)


def run(args):
    # model
    net = EnDeCoder()
    log(args, net)

    # loss
    if args.loss == "BCE":
        loss = nn.BCELoss()
    elif args.loss == "ABC":
        loss = nn.optim.ABCLoss()
    

    # optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise("Only adam optimizer is supported right now!!!!!")
    
    # training/validating loop
    for epoch in range(args.epochs):
        if epoch%args.valrate == 0:
            loop(optimizer, loss, net, epoch, train=False)
        else:
            loop(optimizer, loss, net, epoch, train=True)



def loop(optimizer, loss_fn, net, epoch, train=True):
    data_loader = load_data()
    epoch_loss = 0
    epoch_acc = 0

    for x, y in tqdm(data_loader):
        
        if not train:
            with torch.no_grad():
                output = net(x)
        else: output = net(x)

        loss = loss_fn(output, y)
        epoch_loss += loss
        
        if train:
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()

        # calculate the accuracy
        acc = 1
        epoch_acc +=acc

    epoch_loss = epoch_loss/(len(data_loader)/args.batch_size)
    epoch_acc = epoch_acc/(len(data_loader)/args.batch_size)
    
    if train:
        info = "Train "
    else:
        info = "Val "
    
    info = info + f"Epoch {epoch}/{args.epochs}  Acc: {acc}, loss: {loss} "
    log(args, info)



            


    

def load_data(args, dataset):

    dl = DataLoader(dataset, args.batch_size, num_workers=0)
    

if __name__ == "__main__":

    args = get_args()
    log(args)

    path = "aklsjd;fka'g"
    dataset = KPEAction(data_path = args.data)
    run(dataset, lr=args.lr, batch_size=args.batch_size)

