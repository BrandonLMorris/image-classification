from absl import app, flags
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

FLAGS = flags.FLAGS
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(_):
    tfms = [ToTensor()]
    trn_ds = FashionMNIST(FLAGS.dataset_path, download=FLAGS.download,
                          transform=Compose(tfms))
    tst_ds = FashionMNIST(FLAGS.dataset_path, download=FLAGS.download,
                          train=False, transform=Compose(tfms))
    trn = DataLoader(trn_ds, FLAGS.batch_size, shuffle=True)
    tst = DataLoader(tst_ds, FLAGS.batch_size, shuffle=False)
    model = FMNISTModel().to(dev)
    for epoch in range(FLAGS.epochs):
        print(f'Starting epoch {epoch+1}')
        loss = train(model, trn, tst)
        print(f'\tAverage loss of {loss:.2f}')
        acc = eval(model, trn)
        print(f'\tAccuracy of {acc*100:.2f}%')


def train(model, trn, test):
    model.train()
    losses = []
    for inp, targ in trn:
        model.optimizer.zero_grad()
        out = model(inp.to(dev))
        loss = model.criterion(out, targ.to(dev))
        loss.backward()
        model.optimizer.step()
        losses.append(loss)
    losses = torch.tensor(losses)
    return torch.mean(losses)


def eval(model, dl):
    model.eval()
    count, correct = float(len(dl.dataset)), torch.tensor(0).to(dev)
    for inp, targ in dl:
        out = torch.argmax(model(inp.to(dev)), dim=1)
        correct.add_(out.eq(targ.to(dev)).sum())
    correct = correct.cpu().float()
    return correct/count


class FMNISTModel(nn.Module):
    def __init__(self):
        super(FMNISTModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Linear(32, 10)

        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), 3e-3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=1)


if __name__ == '__main__':
    flags.DEFINE_string('dataset_path', '/hdd/datasets/fashion-mnist',
                     'Path to the dataset')
    flags.DEFINE_integer('batch_size', 64, 'The batch size')
    flags.DEFINE_integer('epochs', 20, 'The number of epochs to train for')
    flags.DEFINE_bool('download', False, 'If true, downloads the dataset')
    app.run(main)

