import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from util.visualizer import Visualizer


from models import networks



from data.unaligned_dataset import UnalignedDataset


class ClassifierDataset(UnalignedDataset):
    def __init__(self, opt):
        UnalignedDataset.__init__(self, opt)

    def __getitem__(self, index):
        data = UnalignedDataset.__getitem__(self, index)
        return data['A'], data['B']

    '''
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return A, B
    '''

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    arg_str = "Args: " + ', '.join(f'{k}={v}' for k, v in vars(opt).items())
    print(arg_str)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    #dataset = ClassifierDataset(opt)
    #dataset =  torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)  # Single threaded dataloader

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)  

    val_opt = opt
    val_opt.phase = 'test'      # Select data from test folder for validation
    val_opt.serial_batches = True   # No shuffling
    val_dataset = create_dataset(val_opt)
    #val_dataset = ClassifierDataset(val_opt)
    #val_dataset =  torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)  # Single threaded dataloader
    
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    dataset_size = len(val_dataset)    # get the number of images in the dataset.
    print('The number of validation images = %d' % dataset_size)

    domain_disc = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, gpu_ids=[], q=opt.q, is_residual=opt.is_residual, use_bias=not opt.no_bias)
    domain_disc.to(device)

    loss_func = networks.GANLoss(opt.gan_mode, target_real_label=1.0, target_fake_label=0.0).to(device)    # Use standard GAN loss for training
    optimizer = torch.optim.Adam(domain_disc.parameters(), lr=opt.lr)
    best_val_loss = 1000

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        
        train_loss = []
        val_loss = []

        domain_disc.train()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            data_A = data['A'].to(device)
            data_B = data['B'].to(device)

            #print(data_A.shape, data_B.shape)

            outputA = domain_disc(data_A) 
            outputB = domain_disc(data_B) 

            loss = loss_func(outputA, False) + loss_func(outputB, True)     # False for source distribution, True for target distribution
            train_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        correct = 0
        total = 0
        domain_disc.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset):  # inner loop within one epoch
                data_A = data['A'].to(device)
                data_B = data['B'].to(device)

                outputA = domain_disc(data_A)
                outputB = domain_disc(data_B)

                classifications_A = torch.mean(outputA, dim=[1,2,3])
                classifications_B = torch.mean(outputB, dim=[1,2,3])

                correct += torch.sum(torch.where(classifications_A <= 0.5, 1, 0)) + torch.sum(torch.where(classifications_B > 0.5, 1, 0)) 
                total += outputA.shape[0] + outputB.shape[0]

                loss = loss_func(outputA, False) + loss_func(outputB, True)     # False for source distribution, True for target distribution
                val_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)
        val_loss = sum(val_loss) / len(val_loss)
        val_acc = float(correct / total)

        if val_loss < best_val_loss:              # cache our model every <save_epoch_freq> epochs
            print(f'New best validation loss {val_loss}, saving the model at the end of epoch {epoch}.')
            torch.save(domain_disc.cpu().state_dict(), "domain_discriminator_model.pth.tar")
            best_val_loss = val_loss
            domain_disc.to(device)

        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} | train loss {round(train_loss, 5)} | val loss {round(val_loss, 5)} | val acc {round(val_acc, 4)} | Time Taken: {round(time.time() - epoch_start_time, 1)}s')
